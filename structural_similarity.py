import re
from typing import List, Tuple, Dict, Optional, Union, TypeAlias
import word_embeddings
import numpy as np
import pandas as pd

# Type aliases for better readability
Relation: TypeAlias = Tuple[str, str]  # (node_id, relation_type)
Relationship: TypeAlias = List[Union[List[str], str]]  # [internal_words, relation, external_words]

class StructuralSimilarity:
    """
    Calculate structural similarity between two DRG relationships.
    """
    
    def __init__(self, weights: Optional[Dict[str, float]] = None):
        # Weights as stated in the article
        self.weights = weights or {
            'Actor': 8,
            'Theme': 8, 
            'experiencer': 6,
            'is': 4,
            'in': 3,
            'other': 1
        }
        
    @staticmethod
    def get_relations(drg: str) -> List[Relation]:
        """
        Extract relations and roles from DRG text.
        
        Args:
            drg: Input DRG format text
            
        Returns:
            List of tuples containing (node_id, relation_type)
                Es: [('c3', 'Actor'), ('c6', 'Theme'), ('c12', 'equality')]
        """
        relations = re.findall(r'k.(?::p1)? (?:relation|role) (c\d+):(\w+)', drg)
        
        # Catch the equality relations corresponding to is relations
        is_relations = re.findall(r'(c\d+):equality .+ \[ (is) \]', drg)

        # Replace "equality" with "is"
        # This is needed because "is" relations in the article, which are a type of 
        # "equality" relation, are considered differently from other "equality" relations
        for rel in is_relations:
            relations.remove((rel[0], 'equality'))
            relations.append((rel[0], rel[1]))
            
        return relations

    @staticmethod
    def get_instance(instance: str, drg: str) -> str:
        """
        Get the word corresponding to an instance ID.
        
        Args:
            instance: Instance ID to look up (e.g. "x5")
            drg: DRG text to search in
            
        Returns:
            Corresponding word (e.g. "hill")
        """
        # The word following the ID and the word between [ ] may differ, the former is the word lemma (the basic form).
        # e.g.: "high" and "higher", "surround" and "surrounds"
        
        # If the input word is an instance, get the word following the ID
        tmp = re.findall(fr'c..?:(\w+)\.?(?:[:-]+\d+)? (?:instance) k.:(?:p1:)?{instance}', drg)
        
        if not tmp:
            # If the word is an arg, get the word between [ ], so the corresponding word from the text
            tmp = re.findall(fr'(?:c..?)?:\w+\.?(?:[:-]+\d+)? (?:arg|referent) k.:(?:p1:)?{instance} \d+ \[ ?(\w+) ?]', drg)
            
        return tmp[0]
    
    @staticmethod
    def get_relationships(relations: List[Relation], drg: str) -> List[Relationship]:
        """
        Convert relations and roles into relationships.
        
        Args:
            relations: List of relation tuples (node_id, relation_type)
                E.g.: [('c3', 'Actor'), ('c6', 'Theme'), ('c12', 'equality')]
            drg: DRG text
            
        Returns:
            List of [internal_word, relation_type, external_word]
                E.g.: [['area', 'of', 'land'], ['that', 'Actor', 'surround'], ['that', 'Theme', 'higher'], ['hill', 'is', 'area']]
        """
        relationships = []
        for relation in relations:
            id = relation[0]
            tmp = re.findall(
                fr'{id}:(\w+)(?:[:-]+\d)? (int|ext) k.:(?:p1:)?([xes].)', 
                drg
            )
            relationships.append([
                StructuralSimilarity.get_instance(tmp[1][2], drg),
                relation[1],
                StructuralSimilarity.get_instance(tmp[0][2], drg)
            ])
        return relationships
        
    @staticmethod
    def eliminate_duplicates(relationships: List[Relationship]) -> List[Relationship]:
        """
        Remove duplicate relationships.
        
        Args:
            relationships: List of [internal_word, relation_type, external_word]
            
        Returns:
            List of [internal_word, relation_type, external_word]
        """
        relationships = set(tuple(rel) for rel in relationships)
        return [list(rel) for rel in relationships]
    
    @staticmethod
    def get_expansions(relationships: List[Relationship]) -> List[Relationship]:
        """
        Expand the words that are part of the “is” or “equality” relations, as briefly indicated in the article.
        The resulting relationships will be between Lists of words
        
        Args:
            relationships: List of [internal_word, relation_type, external_word]
            
        Returns:
            List of [internal_words, relation_type, external_words] where internal_words and external_words are Lists
        """
        expanded_relationships = []
        word_expansions = {}    # Dict to manage the expansions

        for rel in relationships:
            if rel[1] == "equality" or rel[1] == "is":
                if rel[0] not in word_expansions:
                    word_expansions[rel[0]] = []
                if rel[2] not in word_expansions:
                    word_expansions[rel[2]] = []
                word_expansions[rel[0]].append(rel[2])
                word_expansions[rel[2]].append(rel[0])

        for rel in relationships:
            expanded_interior = [rel[0]] + word_expansions.get(rel[0], [])
            expanded_exterior = [rel[2]] + word_expansions.get(rel[2], [])
            expanded_relationships.append([expanded_interior, rel[1], expanded_exterior])

        return expanded_relationships
    
    @staticmethod
    def get_max_similarity(list1: List[str], list2: List[str]) -> float:
        """
        Calculate the maximum similarity between all elements of two word lists.
        
        Args:
            list1: First list of words
            list2: Second list of words
            
        Returns:
            Maximum similarity score
        """
        return max(word_embeddings.get_word_similarity(a, b) for a in list1 for b in list2)

    @staticmethod
    def get_relationship_similarity(relationship1: List[Relationship], relationship2: List[Relationship]) -> float:
        """Calculate similarity between two relationships.
        
        Args:
            relationship1: First relationship [internal words, relation, external words]
            relationship2: Second relationship [internal words, relation, external words]
            
        Returns:
            Similarity score between the relationships
        """
        [int1, rel_name1, ext1] = relationship1
        [int2, rel_name2, ext2] = relationship2
        
        # Calculate internal words similarity
        int_sim = StructuralSimilarity.get_max_similarity(int1, int2) if len(int1) > 1 or len(int2) > 1 else word_embeddings.get_word_similarity(int1[0], int2[0])
        
        # Calculate external words similarity  
        ext_sim = StructuralSimilarity.get_max_similarity(ext1, ext2) if len(ext1) > 1 or len(ext2) > 1 else word_embeddings.get_word_similarity(ext1[0], ext2[0])
        
        # Calculate similarity between relations as stated in the article
        if rel_name1 == rel_name2:
            name_sim = 1
        elif rel_name1 in ['is', 'equality'] and rel_name2 in ['is', 'equality']:
            name_sim = 0.7
        else:
            name_sim = 0.73
        
        return ((int_sim + ext_sim) / 2) * name_sim     # Formula stated in the article

    @staticmethod
    def get_relation_matrix(relationships1: List[Relationship], relationships2: List[Relationship]) -> pd.DataFrame:
        """
        Calculate the relation matrix which contains the similarity between the relations.
        
        Args:
            relationship1: First relationship [internal words, relation, external words]
            relationship2: Second relationship [internal words, relation, external words]
            
        Returns:
            Relation matrix as a Pandas DataFrame
        """
        rows = len(relationships1)
        cols = len(relationships2)
        rel_matrix = np.zeros((rows, cols))
        
        # Populate the relation matrix with the relationship similarities
        for i in range(rows):
            for j in range(cols):
                rel_matrix[i][j] = StructuralSimilarity.get_relationship_similarity(
                    relationships1[i], 
                    relationships2[j]
                )

        # Get the relation matrix as a Pandas DataFrame, with relation names as indexes
        row_index = [similarity.toString(relationships1[i]) for i in range(rows)]     
        col_index = [similarity.toString(relationships2[j]) for j in range(cols)]
        
        print(row_index, col_index)
        return pd.DataFrame(rel_matrix, index=row_index, columns=col_index)

    def get_structural_similarity(self, df: pd.DataFrame) -> float:
        max_sims = df.max(axis=1)   # Get max similarity for every row
        sim_val = 0
        weights_sum = 0
        for i in range(len(max_sims)):
            relation = max_sims.index[i].split()[1]
            
            # If relation is unknown
            if(relation not in self.weights):
                relation = "other"

            # Formula from the paper
            weights_sum += self.weights[relation]
            sim_val += max_sims[i] * self.weights[relation]

        return sim_val/weights_sum

    @staticmethod
    def toString(rel: str) -> str:
        return str(rel[0]) + " " + str(rel[1]) + " " + str(rel[2])

    @staticmethod
    def read_drg(path: str) -> str:
        with open(path, "r") as f1:
            drg = f1.read()
            
        return drg

# Initialization
word_embeddings = word_embeddings.WordEmbeddings()  # loads the word embedding model
similarity = StructuralSimilarity()

# DRG paths and reading
path = "parser/candc/working/"
drg_name1 = "hill.drg"
drg_name2 = "mound.drg"

drg1 = similarity.read_drg(drg_name1)
drg2 = similarity.read_drg(drg_name2)
    
# Relations and relationships extractions
relations1 = similarity.get_relations(drg1)  # [('c3', 'Actor'), ...]
relations2 = similarity.get_relations(drg2)

relationships1 = similarity.get_relationships(relations1, drg1)  # [['hill', 'is', 'high'], ...]
relationships2 = similarity.get_relationships(relations2, drg2)

relationships1 = similarity.eliminate_duplicates(relationships1)
relationships2 = similarity.eliminate_duplicates(relationships2)

ext_relationships1 = similarity.get_expansions(relationships1)
ext_relationships2 = similarity.get_expansions(relationships2)

# Relation matrix generation
rel_matrix_df = similarity.get_relation_matrix(ext_relationships1, ext_relationships2)

print(rel_matrix_df)
print("Similarity between sentence1 and sentence2", similarity.get_structural_similarity(rel_matrix_df))
print("Similarity between sentence2 and sentence1", similarity.get_structural_similarity(rel_matrix_df))


"""
Implementation based on:
Mamdouh Farouk, Measuring text similarity based on structure and word embedding, Cognitive Systems Research, 2020.
"""