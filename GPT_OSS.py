from openai import OpenAI
import re
import random

from rdkit import Chem

MINIMUM = 1e-10

class gpt_oss:
    def __init__(self):

        self.gpt = OpenAI(
            base_url = 'http://10.34.35.194:11434/v1',
            api_key='ollama', # required, but unused
        )
        
        self.requirements = """\n\nYour output should follow the format: {<<<Explaination>>>: $EXPLANATION, <<<Molecule>>>: \\box{$Molecule}}. Here are the requirements:\n
        \n\n1. $EXPLANATION should be your analysis.\n2. The $Molecule should be the smiles of your propsosed molecule.\n3. The molecule should be valid.
        """

        self.task2description = {
                'qed': 'I have two molecules and their QED scores. The QED score measures the drug-likeness of the molecule.\n\n',
                'jnk3': 'I have two molecules and their JNK3 scores. The JNK3 score measures a molecular\'s biological activity against JNK3.\n\n',
                'drd2': 'I have two molecules and their DRD2 scores. The DRD2 score measures a molecule\'s biological activity against a biological target named the dopamine type 2 receptor (DRD2).\n\n',
                'gsk3b': 'I have two molecules and their GSK3$\beta$ scores. The GSK3$\beta$ score measures a molecular\'s biological activity against Glycogen Synthase Kinase 3 Beta.\n\n',
                'isomers_C9H10N2O2PF2Cl': 'I have two molecules and their isomer scores. The isomer score measures a molecule\'s similarity in terms of atom counter to C9H10N2O2PF2Cl.\n\n',
                'perindopril_mpo': 'I have two molecules and their perindopril multiproperty objective scores. The perindopril multiproperty objective score measures the geometric means of several scores, including the molecule\'s Tanimoto similarity to perindopril and number of aromatic rings.\n\n',
                'sitagliptin_mpo': 'I have two molecules and their sitagliptin multiproperty objective scores. The sitagliptin rediscovery score measures the geometric means of several scores, including the molecule\'s Tanimoto similarity to sitagliptin, TPSA score, LogP score and isomer score with C16H15F6N5O.\n\n',
                'ranolazine_mpo': 'I have two molecules and their ranolazine multiproperty objective scores. The ranolazine multiproperty objective score measures the geometric means of several scores, including the molecule\'s Tanimoto similarity to ranolazine, TPSA score LogP score and number of fluorine atoms.\n\n',
                'thiothixene_rediscovery': 'I have two molecules and their thiothixene rediscovery measures a molecule\'s Tanimoto similarity with thiothixene\'s SMILES to check whether it could be rediscovered.\n\n',
                'mestranol_similarity': 'I have two molecules and their mestranol similarity scores. The mestranol similarity score measures a molecule\'s Tanimoto similarity with Mestranol.\n\n',
                }
        self.task2objective = {
                'qed': 'Please propose a new molecule that has a higher QED score. You can either make crossover and mutations based on the given molecules or just propose a new molecule based on your knowledge.\n\n',
                'jnk3': 'Please propose a new molecule that has a higher JNK3 score. You can either make crossover and mutations based on the given molecules or just propose a new molecule based on your knowledge.\n\n',
                'drd2': 'Please propose a new molecule that has a higher DRD2 score. You can either make crossover and mutations based on the given molecules or just propose a new molecule based on your knowledge.\n\n',
                'gsk3b': 'Please propose a new molecule that has a higher GSK3$\beta$ score. You can either make crossover and mutations based on the given molecules or just propose a new molecule based on your knowledge.\n\n',
                'isomers_C9H10N2O2PF2Cl': 'Please propose a new molecule that has a higher isomer score. You can either make crossover and mutations based on the given molecules or just propose a new molecule based on your knowledge.\n\n',
                'perindopril_mpo': 'Please propose a new molecule that has a higher perindopril multiproperty objective score. You can either make crossover and mutations based on the given molecules or just propose a new molecule based on your knowledge.\n\n',
                'sitagliptin_mpo': 'Please propose a new molecule that has a higher sitagliptin multiproperty objective score. You can either make crossover and mutations based on the given molecules or just propose a new molecule based on your knowledge.\n\n',
                'ranolazine_mpo': 'Please propose a new molecule that has a higher ranolazine multiproperty objective score. You can either make crossover and mutations based on the given molecules or just propose a new molecule based on your knowledge.\n\n',
                'thiothixene_rediscovery': 'Please propose a new molecule that has a higher thiothixene rediscovery score. You can either make crossover and mutations based on the given molecules or just propose a new molecule based on your knowledge.\n\n',
                'mestranol_similarity': 'Please propose a new molecule that has a higher mestranol similarity score. You can either make crossover and mutations based on the given molecules or just propose a new molecule based on your knowledge.\n\n',
                }
        
    def sanitize_smiles(self, smi):
        """
        Return a canonical smile representation of smi 
        """
        if smi == '':
            return None
        try:
            mol = Chem.MolFromSmiles(smi, sanitize=True)
            smi_canon = Chem.MolToSmiles(mol, isomericSmiles=False, canonical=True)
            return smi_canon
        except:
            return None
        
    def gpt_request(self, question, temperature=0.0):
        message = [{"role": "system", "content": "You are a helpful agent who can answer the question based on your molecule knowledge."}]

        message.append({"role": "user", "content": question})

        params = {
            "model": "gpt-oss:20b",
            "max_tokens": 2048,
            "temperature": temperature,
            "messages": message
        }

        for retry in range(3):
            try:
                response = self.gpt.chat.completions.create(**params).choices[0].message.content
                message.append({"role": "assistant", "content": response})
                break
            except Exception as e:
                print(f"{type(e).__name__} {e}")

        return message, response

    def edit_smi(self, parent_info: str, task):
        task_definition = self.task2description[task]
        task_objective = self.task2objective[task]
        while(True):
            try:
                prompt = task_definition + parent_info + task_objective + self.requirements

                message, response = self.gpt_request(self.gpt,prompt)
                proposed_smiles = re.search(r'\\box\{(.*?)\}', response).group(1)
                proposed_smiles = self.sanitize_smiles(proposed_smiles)
                
                if proposed_smiles is not None :  return proposed_smiles

            except Exception as e:
                print( '\033[31m' + "Error Invalid Response!! Retry !!" + '\033[0m' )

class GPT_OSS:
    def __init__(self, args):

        self.args = args
        self.LLM = gpt_oss()

    def generational_shift(self, mating_list: list):
        families = []
        for i in range(self.args.offspring_size):
            
            parent = []
            parent.append(random.choice(mating_list))
            parent.append(random.choice(mating_list))

            parent_info = ''
            for j in range(2):
                parent_info += '\n[' + parent[j].smi + ',' + str(parent[j].score) + ']'

            edited_smi = self.LLM.edit_smi(parent_info, self.args.task)
            families.append((edited_smi,parent[0].smi,parent[1].smi))

        return families

