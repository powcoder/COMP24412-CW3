https://powcoder.com
代写代考加微信 powcoder
Assignment Project Exam Help
Add WeChat powcoder
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from inspect import CO_ASYNC_GENERATOR
from itertools import count
from typing import Any, List, Dict
import numpy as np

from learning.util import Algorithm, AlgorithmRegistry

Example = Dict[str, Any]
Examples = List[Example]

from logging import getLogger

logger = getLogger(__name__)


@dataclass(frozen=True)
class AttrLogicExpression(ABC):
    """
    Abstract base class representing a logic expression.
    """
    ...


    @abstractmethod
    def __call__(self, *args, **kwargs):
        ...


@dataclass(frozen=True)
class Conjunction(AttrLogicExpression):
    """
    A configuration of attribute names and the values the attributes should take for this conjunction to evaluate
    to true.

    `attribute_confs` is a map from attribute names to their values.
    """
    attribute_confs: Dict[str, Any]

    def __post_init__(self):
        assert 'target' not in self.attribute_confs, "Nice try, but 'target' cannot be part of the hypothesis."

    def __call__(self, example: Example):
        """
        Evaluates whether the conjunction applies to an example or not. Returns true if it does, false otherwise.


        Args:
            example: Example to check if the conjunction applies.

        Returns:
            True if the values of *all* attributes mentioned in the conjunction and appearing in example are equal,
            false otherwise.


        """
        return all(self.attribute_confs[k] == example[k] for k in set(self.attribute_confs).intersection(example))

    def __repr__(self):
        return " AND ".join(f"{k} = {v}" for k, v in self.attribute_confs.items())


@dataclass(frozen=True)
class Disjunction(AttrLogicExpression):
    """
    Disjunction of conjunctions.
    """
    conjunctions: List[Conjunction]

    def __call__(self, example: Example):
        """
        Evaluates whether the disjunction applies to a given example.

        Args:
            example: Example to check if the disjunction applies.

        Returns: True if any of its conjunctions returns true, and false if none evaluates to true.

        """
        return any(c(example) for c in self.conjunctions)

    def __repr__(self):
        return " " + "\nOR\n ".join(f"{v}" for v in self.conjunctions)


class Tree(ABC):
    """
    This is an abstract base class representing a leaf or a node in a tree.
    """
    ...


@dataclass
class Leaf(Tree):
    """
    This is a leaf in the tree. It's value is the (binary) classification, either True or False.
    """
    target: bool


@dataclass
class Node(Tree):
    """
    This is a node in the tree. It contains the attribute `attr_name` which the node is splitting on and a dictionary
    `branches` that represents the children of the node and maps from attribute values to their corresponding subtrees.
    """
    attr_name: str
    branches: Dict[Any, Tree] = field(default_factory=dict)


def same_target(examples: Examples) -> bool:
    """
    This function checks whether the examples all have the same target.

    Args:
        examples: Observations to check

    Returns: Whether the examples all have the same target.
    """
    for example in examples:
        if example['target'] != examples[0]['target']:
            return False
    return True
    raise NotImplementedError()


def plurality_value(examples: Examples) -> bool:
    """
    This function returns whether there are more positively or negatively classified examples in the dataset.
    Args:
        examples: Examples to check.

    Returns: True if more examples classified as positive, False otherwise.

    """
    count_positive = 0
    count_negative = 0
    for example in examples:
        if example['target'] == True:
            count_positive+=1
        else:
            count_negative+=1
    if count_positive > count_negative:
        return True
    else:
        return False

    raise NotImplementedError()


def binary_entropy(examples: Examples) -> float:
    """
    Calculates the binary (shannon) entropy of a dataset regarding its classification.
    Args:
        examples: Dataset to calculate the shannon entropy.

    Returns: The shannon entropy of the classification of the dataset.

    """
    count_positive = 0
    count_negative = 0
    for example in examples:
        if example['target'] == True:
            count_positive+=1
        else:
            count_negative+=1
    if count_positive == 0 or count_negative == 0:
        return 0
    else:
        return -(count_positive/len(examples))*np.log2(count_positive/len(examples))-(count_negative/len(examples))*np.log2(count_negative/len(examples))
    raise NotImplementedError()

def to_logic_expression(tree: Tree) -> AttrLogicExpression:
    """
    Converts a Decision tree to its equivalent logic expression.
    Args:
        tree: Tree to convert.

    Returns: The corresponding logic expression consisting of attribute values, conjunctions and disjunctions.

    """
    if isinstance(tree, Leaf):
        return Leaf(tree.target)
    elif isinstance(tree, Node):
        Conjunction.attribute_confs={tree.attr_name: tree.branches.keys()}
        return Disjunction([to_logic_expression(branch) for branch in tree.branches.values()])
    
    # raise NotImplementedError()


@AlgorithmRegistry.register("dtl")
class DecisionTreeLearner(Algorithm):
    """
    This is the decision tree learning algorithm.
    """

    def find_hypothesis(self) -> AttrLogicExpression:
        tree = self.decision_tree_learning(examples=self.dataset.examples, attributes=self.dataset.attributes,
                                           parent_examples=[])
        return to_logic_expression(tree)

    def decision_tree_learning(self, examples: Examples, attributes: List[str], parent_examples: Examples) -> Tree:
        """
        This is the main function that learns a decision tree given a list of example and attributes.
        Args:
            examples: The training dataset to induce the tree from.
            attributes: Attributes of the examples.
            parent_examples: Examples from previous step.

        Returns: A decision tree induced from the given dataset.
        """
        if len(examples) == 0:
            return Leaf(target=plurality_value(parent_examples))
        elif same_target(examples):
            return Leaf(target=examples[0]['target'])
        elif len(attributes) == 0:
            return Leaf(target=plurality_value(examples))
        else:
            best_attribute = self.get_most_important_attribute(attributes,examples)
            tree = Node(attr_name=best_attribute)
            for value in examples[best_attribute]:
                new_examples = [e for e in examples if e[best_attribute] == value]
                new_attributes = [a for a in attributes if a != best_attribute]
                tree.branches[value] = self.decision_tree_learning(new_examples, new_attributes, examples)
            return tree

    def get_most_important_attribute(self, attributes: List[str], examples: Examples) -> str:
        """
        Returns the most important attribute according to the information gain measure.
        Args:
            attributes: The attributes to choose the most important attribute from.
            examples: Dataset from which the most important attribute is to be inferred.

        Returns: The most informative attribute according to the dataset.

        """
        if len(attributes) == 0:
            return None
        else:
            max_gain = -1
            best_attribute = None
            for attribute in attributes:
                gain = self.information_gain(examples,attribute)
                if gain > max_gain:
                    max_gain = gain
                    best_attribute = attribute
            return best_attribute

        
        raise NotImplementedError()

    def information_gain(self, examples: Examples, attribute: str) -> float:
        """
        This method calculates the information gain (as presented in the lecture)
        of an attribute according to given observations.

        Args:
            examples: Dataset to infer the information gain from.
            attribute: Attribute to infer the information gain for.

        Returns: The information gain of the given attribute according to the given observations.
    
        """
        # Get probability of each value of the attribute
        count_positive = 0
        count_negative = 0
        for example in examples:
            if example[attribute] == True:
                count_positive+=1
            else:
                count_negative+=1
        prob_positive = count_positive/len(examples)
        prob_negative = count_negative/len(examples)

        # Get probability of each value of the attribute given the target
        count_T_given_T = 0
        count_T_given_F = 0
        count_F_given_T = 0
        count_F_given_F = 0
        for example in examples:
            if example[attribute] == True:
                if example['target'] == True:
                    count_T_given_T+=1
                else:
                    count_T_given_F+=1
            else:
                if example['target'] == True:
                    count_F_given_T+=1
                else:
                    count_F_given_F+=1
        prob_count_T_given_T = count_T_given_T/count_positive
        prob_count_T_given_F = count_T_given_F/count_positive
        prob_count_F_given_T = count_F_given_T/count_negative
        prob_count_F_given_F = count_F_given_F/count_negative

        # Calculate the information gain
        H_attribute_T = -(prob_count_T_given_T*np.log2(prob_count_T_given_T) + prob_count_T_given_F*np.log2(prob_count_T_given_F))
        H_attribute_F = -(prob_count_F_given_T*np.log2(prob_count_F_given_T) + prob_count_F_given_F*np.log2(prob_count_F_given_F))

        if prob_count_T_given_T == 0:
            H_attribute_T = -prob_count_T_given_F*np.log2(prob_count_T_given_F)
        elif prob_count_T_given_F ==0:
            H_attribute_T = -prob_count_T_given_T*np.log2(prob_count_T_given_T)
        elif prob_count_F_given_T == 0:
            H_attribute_F = -prob_count_F_given_F*np.log2(prob_count_F_given_F)
        elif prob_count_F_given_F == 0:
            H_attribute_F = -prob_count_F_given_T*np.log2(prob_count_F_given_T)

        H_attribute = prob_positive*H_attribute_T + prob_negative*H_attribute_F
        H_target = binary_entropy(examples)
        return H_target - H_attribute
        raise NotImplementedError()


@AlgorithmRegistry.register("your-algo-name")
class MyDecisionTreeLearner(DecisionTreeLearner):
    ...
    