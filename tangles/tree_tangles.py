from pathlib import Path
from typing import Optional
import matplotlib.pyplot as plt

import networkx as nx
from networkx.drawing.nx_pydot import graphviz_layout

import bitarray as ba
import numpy as np
from PIL import Image
from .plotting import plot_soft_predictions

from .tangles import Tangle, core_algorithm
from .utils import compute_hard_predictions, matching_items, Orientation, normalize
from .cost_functions import BipartitionSimilarity
from .data_types import Cuts

MAX_CLUSTERS = 50


class TangleNode(object):
    def __init__(
        self,
        parent,
        right_child,
        left_child,
        is_left_child,
        splitting,
        did_split,
        last_cut_added_id,
        last_cut_added_orientation,
        tangle: Tangle,
    ):

        self.parent = parent
        self.right_child = right_child
        self.left_child = left_child
        self.is_left_child = is_left_child

        self.splitting = splitting

        self.did_split = did_split
        self.last_cut_added_id = last_cut_added_id
        self.last_cut_added_orientation = last_cut_added_orientation

        self.tangle = tangle

    @property
    def last_cut_added(self) -> np.ndarray:
        cut = self.tangle.get_cuts().get_cut_at(self.last_cut_added_id)
        if self.last_cut_added_orientation:
            return cut
        else:
            return ~cut

    def __str__(self, height=0):  # pragma: no cover

        if self.parent is None:
            string = "Root"
        else:
            padding = " "
            string = "{}{} -> {}".format(
                padding * height,
                self.last_cut_added_id,
                self.last_cut_added_orientation,
            )

        if self.left_child is not None:
            string += "\n"
            string += self.left_child.__str__(height=height + 1)
        if self.right_child is not None:
            string += "\n"
            string += self.right_child.__str__(height=height + 1)

        return string

    def is_leaf(self):
        return self.left_child is None and self.right_child is None


class ContractedTangleNode(TangleNode):
    def __init__(self, parent, node):

        attributes = node.__dict__
        super().__init__(**attributes)

        self.parent = parent
        self.right_child = None
        self.left_child = None

        self.characterizing_cuts = None
        self.characterizing_cuts_left = None
        self.characterizing_cuts_right = None

        self.is_left_child_deleted = False
        self.is_right_child_deleted = False

        self.p = None
        self.image = None

    def get_characterizing_cut_values(
        self, characterizing_cuts: dict[int, Orientation]
    ) -> dict[int, np.ndarray]:
        """
        Returns the values of the cuts in the characterizing cuts dictionary, with the orientation.
        IDs are changed to IDs of the unsorted cuts.
        """
        ret = {}
        own_cuts: Cuts = self.tangle._cuts
        for cut_id, orientation in characterizing_cuts.items():
            cut = orientation.orient_cut(
                own_cuts.get_cut_at(cut_id, from_unsorted=False)
            )
            ret[own_cuts.unsorted_id(cut_id)] = cut
        return ret

    def __repr__(self) -> str:
        return "Node: " + self.__str__()

    def __str__(self) -> str:
        if self.parent is None:
            return "Root"
        else:
            orientation = "T" if self.last_cut_added_orientation else "F"
            return f"{self.last_cut_added_id}" + f"{orientation}"

    def to_string_tree_like(self, height: int = 0):
        string = ""

        if self.parent is None:
            string += "Root\n"

        padding = "  "
        string_cuts = (
            ["{} -> {}".format(k, v) for k, v in self.characterizing_cuts_left.items()]
            if self.characterizing_cuts_left is not None
            else ""
        )
        string += "{}{} left: {}\n".format(
            padding * height, self.last_cut_added_id, string_cuts
        )

        string_cuts = (
            ["{} -> {}".format(k, v) for k, v in self.characterizing_cuts_right.items()]
            if self.characterizing_cuts_right is not None
            else ""
        )
        string += "{}{} right: {}\n".format(
            padding * height, self.last_cut_added_id, string_cuts
        )

        if self.left_child is not None:
            string += "\n"
            string += self.left_child.to_string_tree_like(height=height + 1)
        if self.right_child is not None:
            string += "\n"
            string += self.right_child.to_string_tree_like(height=height + 1)

        return string


# created new TangleNode and adds it as child to current node
def _add_new_child(
    current_node, tangle, last_cut_added_id, last_cut_added_orientation, did_split
):
    new_node = TangleNode(
        parent=current_node,
        right_child=None,
        left_child=None,
        is_left_child=last_cut_added_orientation,
        splitting=False,
        did_split=did_split,
        last_cut_added_id=last_cut_added_id,
        last_cut_added_orientation=last_cut_added_orientation,
        tangle=tangle,
    )

    if new_node.is_left_child:
        current_node.left_child = new_node
    else:
        current_node.right_child = new_node

    return new_node


class TangleTree(object):
    def __init__(self, agreement, cuts, max_clusters=None):

        self.root = TangleNode(
            parent=None,
            right_child=None,
            left_child=None,
            splitting=None,
            is_left_child=None,
            did_split=True,
            last_cut_added_id=-1,
            last_cut_added_orientation=None,
            tangle=Tangle(cuts=cuts),
        )
        self.max_clusters = max_clusters
        self.active = [self.root]
        self.maximals = []
        self.will_split = []
        self.is_empty = True
        self.agreement = agreement

    def __str__(self):  # pragma: no cover
        return str(self.root)

    # function to add a single cut to the tree
    # function checks if tree is empty
    # --- stops if number of active leaves gets too large ! ---
    def add_cut(self, cut, cut_id):
        if self.max_clusters and len(self.active) >= self.max_clusters:
            print("Stopped since there are more then 50 leaves already.")
            return False

        current_active = self.active
        self.active = []

        could_add_one = False
        # Go through all nodes that are on the order of the preceding cut.
        # Check if we can add the current cut to them.
        for current_node in current_active:
            could_add_node, did_split, is_maximal = self._add_children_to_node(
                current_node, cut, cut_id
            )
            could_add_one = could_add_one or could_add_node

            if did_split:
                current_node.splitting = True
                self.will_split.append(current_node)
            elif is_maximal:
                self.maximals.append(current_node)

        if could_add_one:
            self.is_empty = False

        return could_add_one

    def _add_children_to_node(self, current_node, cut, cut_id):
        old_tangle = current_node.tangle

        if cut.dtype is not bool:
            cut = cut.astype(bool)

        # Tangle with the cut added in present orientation.
        new_tangle_true = old_tangle.add(
            new_cut=ba.bitarray(cut.tolist()),
            new_cut_id=cut_id,
            orientation=True,
            min_size=self.agreement,
        )
        # Tangle with the cut added in opposite orientation.
        new_tangle_false = old_tangle.add(
            new_cut=ba.bitarray((~cut).tolist()),
            new_cut_id=cut_id,
            orientation=False,
            min_size=self.agreement,
        )

        could_add_one = False

        # Case of a splitting tangle, we could add both orientations
        if new_tangle_true is not None and new_tangle_false is not None:
            did_split = True
        else:
            did_split = False

        # Cut could not be added in any orientation.
        if new_tangle_true is None and new_tangle_false is None:
            is_maximal = True
        else:
            is_maximal = False

        # Cut could be added in the original orientation.
        if new_tangle_true is not None:
            could_add_one = True
            new_node = _add_new_child(
                current_node=current_node,
                tangle=new_tangle_true,
                last_cut_added_id=cut_id,
                last_cut_added_orientation=True,
                did_split=did_split,
            )
            self.active.append(new_node)

        # Cut could be added in the opposite orientation.
        if new_tangle_false is not None:
            could_add_one = True
            new_node = _add_new_child(
                current_node=current_node,
                tangle=new_tangle_false,
                last_cut_added_id=cut_id,
                last_cut_added_orientation=False,
                did_split=did_split,
            )
            self.active.append(new_node)

        return could_add_one, did_split, is_maximal

    def plot_tree(self, path=None):  # pragma: no cover

        tree = nx.Graph()
        labels = self._add_node_to_nx(tree, self.root)

        pos = graphviz_layout(tree, prog="dot")

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(19, 15))
        nx.draw_networkx(
            tree,
            pos=pos,
            ax=ax,
            labels=labels,
            node_size=30000,
            font_size=70,
            width=5.0,
            node_color=r"#80b9f2",
        )
        if path is not None:
            plt.savefig(path)
        else:
            plt.show()

    def _add_node_to_nx(
        self, tree, node, parent_id=None, direction=None
    ):  # pragma: no cover

        if node.parent is None:
            my_id = "root"
            my_label = "Root"

            if node.image is not None:
                tree.add_node(my_id, image=node.image)
            else:
                tree.add_node(my_id)

        else:
            my_id = parent_id + direction
            str_o = "T" if node.last_cut_added_orientation else "F"
            my_label = "{}{}".format(node.last_cut_added_id, str_o)

            if node.image is not None:
                tree.add_node(my_id, image=node.image)
            else:
                tree.add_node(my_id)
            tree.add_edge(my_id, parent_id)

        labels = {my_id: my_label}

        if node.left_child is not None:
            left_labels = self._add_node_to_nx(
                tree, node.left_child, parent_id=my_id, direction="left"
            )
            labels = {**labels, **left_labels}
        if node.right_child is not None:
            right_labels = self._add_node_to_nx(
                tree, node.right_child, parent_id=my_id, direction="right"
            )
            labels = {**labels, **right_labels}

        return labels


class ContractedTangleTree(TangleTree):

    # noinspection PyMissingConstructor
    def __init__(self, tree):

        self.is_empty = tree.is_empty
        self.processed_soft_prediction = False
        self.maximals = []
        self.splitting = []
        self.root = self._contract_subtree(parent=None, node=tree.root)

    def __str__(self):  # pragma: no cover
        return str(self.root)

    def prune(self, prune_depth=1, verbose=True):
        self._delete_noise_clusters(self.root, depth=prune_depth)
        if verbose:
            print(
                "\t{} clusters after cutting out short paths.".format(
                    len(self.maximals)
                )
            )

    def _delete_noise_clusters(self, node, depth):
        if depth == 0:
            return

        if node.is_leaf():
            if node.parent is None:
                Warning(
                    "This node is a leaf and the root at the same time. This tree is empty!"
                )
            else:
                node_id = node.last_cut_added_id
                parent_id = node.parent.last_cut_added_id

                diff = node_id - parent_id

                if diff <= depth:
                    self.maximals.remove(node)
                    node.parent.splitting = False
                    if node.is_left_child:
                        node.parent.left_child = None
                        node.parent.is_left_child_deleted = True
                    else:
                        node.parent.right_child = None
                        if node.parent.is_left_child_deleted:
                            self.maximals.append(node.parent)
                            self._delete_noise_clusters(node.parent, depth)

        else:
            self._delete_noise_clusters(node.left_child, depth)
            if not node.splitting:
                self.splitting.remove(node)

            self._delete_noise_clusters(node.right_child, depth)

            if not node.splitting:
                if node.parent is None:
                    if node.right_child is not None:
                        self.root = node.right_child
                        self.root.parent = None
                    elif node.left_child is not None:
                        self.root = node.left_child
                        self.root.parent = None
                else:
                    if node in self.splitting:
                        self.splitting.remove(node)
                        if node.is_left_child:
                            node.parent.left_child = node.left_child
                        else:
                            node.parent.right_child = node.left_child
                    else:
                        if node.right_child is not None:
                            if node.is_left_child:
                                node.parent.left_child = node.right_child
                            else:
                                node.parent.right_child = node.right_child

    def calculate_setP(self):
        self._calculate_characterizing_cuts(self.root)

    def _calculate_characterizing_cuts(self, node):

        if node.left_child is None and node.right_child is None:
            node.characterizing_cuts = dict()
            return
        else:
            if node.left_child is not None and node.right_child is not None:
                self._calculate_characterizing_cuts(node.left_child)
                self._calculate_characterizing_cuts(node.right_child)

                process_split(node)
                return

    # As python has no Tail Call Optimization, it is more beneficial to
    # use contract_subtree in an iterative fashion. Else we quickly
    # get in the territory of a stack explosion.
    def _contract_subtree_iterative(self, parent, node):
        current_node = node

        while True:
            if current_node.left_child is None and current_node.right_child is None:
                # is leaf so create new node
                contracted_node = ContractedTangleNode(parent=parent, node=current_node)
                self.maximals.append(contracted_node)
                return contracted_node
            elif (
                current_node.left_child is not None
                and current_node.right_child is not None
            ):
                # is splitting so create new node
                contracted_node = ContractedTangleNode(parent=parent, node=current_node)

                contracted_left_child = self._contract_subtree_iterative(
                    parent=contracted_node, node=current_node.left_child
                )
                contracted_node.left_child = contracted_left_child
                # let it know that it is a left child!
                contracted_node.left_child.is_left_child = True

                contracted_right_child = self._contract_subtree_iterative(
                    parent=contracted_node, node=current_node.right_child
                )
                contracted_node.right_child = contracted_right_child
                # let it know that it is a right child!
                contracted_node.right_child.is_left_child = False

                self.splitting.append(contracted_node)

                return contracted_node
            else:
                if current_node.left_child is not None:
                    current_node = current_node.left_child
                elif current_node.right_child is not None:
                    current_node = current_node.right_child

    def _contract_subtree(self, parent, node):
        return self._contract_subtree_iterative(parent, node)

    def plot_soft_tree(self, data, cut_values, path=None, show=True, names: Optional[list[str]] = None):
        plot_soft_predictions(data, self, cut_values, path=Path("results"), names=names)
        tree = nx.Graph()
        labels = self._add_node_to_nx(tree, self.root)

        pos = graphviz_layout(tree, prog="dot")

        fig, ax = plt.subplots(figsize=(20, 20))

        # see https://networkx.org/documentation/stable/auto_examples/drawing/plot_custom_node_icons.html
        nx.draw_networkx_edges(
            tree,
            pos=pos,
            ax=ax,
            width=5,
            arrows=True,
            arrowstyle="-",
            min_source_margin=30,
            min_target_margin=30,
        )

        # Transform from data coordinates (scaled between xlim and ylim) to display coordinates
        tr_figure = ax.transData.transform
        # Transform from display to figure coordinates
        tr_axes = fig.transFigure.inverted().transform

        # Select the size of the image (relative to the X axis)
        icon_size = 0.30
        icon_center = icon_size / 2.0

        # Add the respective image to each node
        root = True
        for n in tree.nodes:
            xf, yf = tr_figure(pos[n])
            xa, ya = tr_axes((xf, yf))
            # get overlapped axes and plot icon
            if root:
                a = plt.axes(
                    [xa - icon_center - 0.048, ya - icon_center, icon_size, icon_size]
                )
            else:
                a = plt.axes([xa - icon_center, ya - icon_center, icon_size, icon_size])
            a.imshow(Image.open(tree.nodes[n]["image"]))
            a.axis("off")
            root = False
        if path is not None:
            fig.savefig(path, bbox_inches="tight")
        if show is True:
            plt.show()


def process_split(node):
    node_id = node.last_cut_added_id if node.last_cut_added_id else -1

    characterizing_cuts_left = node.left_child.characterizing_cuts
    characterizing_cuts_right = node.right_child.characterizing_cuts

    orientation_left = node.left_child.tangle.get_specification()
    orientation_right = node.right_child.tangle.get_specification()

    # add new relevant cuts
    for id_cut in range(node_id + 1, node.left_child.last_cut_added_id + 1):
        characterizing_cuts_left[id_cut] = Orientation(orientation_left[id_cut])

    for id_cut in range(node_id + 1, node.right_child.last_cut_added_id + 1):
        characterizing_cuts_right[id_cut] = Orientation(orientation_right[id_cut])

    id_not_in_both = (
        characterizing_cuts_left.keys() | characterizing_cuts_right.keys()
    ).difference(characterizing_cuts_left.keys() & characterizing_cuts_right.keys())

    # if cuts are not oriented in both subtrees delete
    for id_cut in id_not_in_both:
        characterizing_cuts_left.pop(id_cut, None)
        characterizing_cuts_right.pop(id_cut, None)

    # characterizing cuts of the current node
    characterizing_cuts = {**characterizing_cuts_left, **characterizing_cuts_right}

    id_cuts_oriented_same_way = matching_items(
        characterizing_cuts_left, characterizing_cuts_right
    )

    # if they are oriented in the same way they are not relevant for distungishing but might be for 'higher' nodes
    # delete in the left and right parts but keep in the characteristics of the current node
    for id_cut in id_cuts_oriented_same_way:
        characterizing_cuts[id_cut] = characterizing_cuts_left[id_cut]
        characterizing_cuts_left.pop(id_cut)
        characterizing_cuts_right.pop(id_cut)

    id_cuts_oriented_both_ways = (
        characterizing_cuts_left.keys() & characterizing_cuts_right.keys()
    )

    # remove the cuts that are oriented in both trees but in different directions from the current node since they do
    # not affect higher nodes anymore
    for id_cut in id_cuts_oriented_both_ways:
        characterizing_cuts.pop(id_cut)

    node.characterizing_cuts_left = characterizing_cuts_left
    node.characterizing_cuts_right = characterizing_cuts_right
    node.characterizing_cuts = characterizing_cuts


def compute_soft_predictions_node(characterizing_cuts, cuts, weight):
    sum_p = np.zeros(len(cuts.values[0]))

    for i, o in characterizing_cuts.items():
        if o.direction == "left":
            sum_p += np.array(cuts.values[i]) * weight[i]
        elif o.direction == "right":
            sum_p += np.array(~cuts.values[i]) * weight[i]

    return sum_p


def compute_soft_predictions_children(node, cuts, weight, verbose=0):
    _, nb_points = cuts.values.shape

    if node.parent is None:
        node.p = np.ones(nb_points)

    if node.left_child is not None and node.right_child is not None:

        unnormalized_p_left = compute_soft_predictions_node(
            characterizing_cuts=node.characterizing_cuts_left, cuts=cuts, weight=weight
        )
        unnormalized_p_right = compute_soft_predictions_node(
            characterizing_cuts=node.characterizing_cuts_right, cuts=cuts, weight=weight
        )

        # normalize the ps
        total_p = unnormalized_p_left + unnormalized_p_right

        p_left = unnormalized_p_left / total_p
        p_right = unnormalized_p_right / total_p

        node.left_child.p = p_left * node.p
        node.right_child.p = p_right * node.p

        compute_soft_predictions_children(
            node=node.left_child, cuts=cuts, weight=weight, verbose=verbose
        )

        compute_soft_predictions_children(
            node=node.right_child, cuts=cuts, weight=weight, verbose=verbose
        )


def tangle_computation(cuts, agreement, verbose):
    """

    Parameters
    ----------
    cuts: cuts
    agreement: int
        The agreement parameter
    verbose:
        verbosity level
    Returns
    -------
    tangles_tree: TangleTree
        The tangle search tree
    """

    if verbose >= 2:
        print("Using agreement = {} \n".format(agreement))
        print("Start tangle computation", flush=True)

    tangles_tree = TangleTree(agreement=agreement, cuts=cuts)
    old_order = None

    unique_orders = np.unique(cuts.costs)

    for order in unique_orders:

        if old_order is None:
            idx_cuts_order_i = np.where(cuts.costs <= order)[0]
        else:
            idx_cuts_order_i = np.where(
                np.all([cuts.costs > old_order, cuts.costs <= order], axis=0)
            )[0]

        if len(idx_cuts_order_i) > 0:

            if verbose >= 2:
                print(
                    "\tCompute tangles of order {} with {} new cuts".format(
                        order, len(idx_cuts_order_i)
                    ),
                    flush=True,
                )

            cuts_order_i = cuts.values[idx_cuts_order_i]
            new_tree = core_algorithm(
                tree=tangles_tree,
                current_cuts=cuts_order_i,
                idx_current_cuts=idx_cuts_order_i,
            )

            if new_tree is None:
                max_order = cuts.costs[-1]
                if verbose >= 2:
                    print("\t\tI could not add any new cuts due to inconsistency")
                    print(
                        "\n\tI stopped the computation at order {} instead of {}".format(
                            old_order, max_order
                        ),
                        flush=True,
                    )
                break
            else:
                tangles_tree = new_tree

                if verbose >= 2:
                    print(
                        "\t\tI found {} tangles of order less or equal {}".format(
                            len(new_tree.active), order
                        ),
                        flush=True,
                    )

        old_order = order

    if tangles_tree is not None:
        tangles_tree.maximals += tangles_tree.active

    if verbose >= 1:
        print(
            "\t{} leaves before cutting out short paths.".format(
                len(tangles_tree.maximals)
            )
        )

    return tangles_tree


def get_hard_predictions(X: np.ndarray, agreement: int, verbose: int = 0):
    """
    Simple function to return hard predictions from a set of cuts X.

    Cuts X are column-wise e.g. each column is a cut.
    """
    verbose_bool = verbose > 0
    cuts = Cuts((X == 1).T)
    cost_function = BipartitionSimilarity(cuts.values.T)
    cuts.compute_cost_and_order_cuts(cost_function, verbose=verbose_bool)

    # Building the tree, contracting and calculating predictions
    tangles_tree = tangle_computation(
        cuts=cuts,
        agreement=agreement,
        # print nothing
        verbose=verbose_bool,
    )

    contracted = ContractedTangleTree(tangles_tree)
    contracted.prune(1, verbose=verbose_bool)

    contracted.calculate_setP()

    # soft predictions
    weight = np.exp(-normalize(cuts.costs))

    compute_soft_predictions_children(
        node=contracted.root, cuts=cuts, weight=weight, verbose=verbose_bool
    )
    contracted.processed_soft_predictions = True

    ys_predicted, _ = compute_hard_predictions(contracted, verbose=verbose_bool)

    return ys_predicted
