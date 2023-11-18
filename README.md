# COMP-7607-FinalProject based on Tk-Instruct

## Motivation
- At the time of the reference paper’s release in April 2022, there were yet any sizable 
datasets of a similar nature for model-development. While the paper found a linear 
growth of model performance with an increase in the number of model size, the 
available training data at the material time was relatively limited leaving a potential 
room for improvement. In response, our group plans to find and use a larger and more 
representative dataset as a form of input ablation variant and to empirically test whether 
training the baseline model with different dataset(s) could lead to improvement on the 
model’s generalization capacity and to analyse the results further for insights2
- Our preliminary research also reveals numerous compelling studies that address the 
same problem (i.e. task diversity) with the use of prompt methods and data 
augmentation techniques. We are exploring ways to incorporate these techniques into 
the baseline model to see if they could lead to a further performance improvement. 
Regarding evaluation metrics, due to the cost of human evaluation, we intend to use 
automatic evaluation at this stage.



## Data Overview

To test only the instruction generalization ability, without the examples, we use only the ‘Definition’ to train and test models.

A task example in Super-NaturalInstructions:
```json
{
  "Contributors": [
      "Swaroop Mishra",
      "Daniel Khashabi"
  ],
  "Source": [
      "quoref"
  ],
  "URL": [
      "https://allenai.org/data/quoref"
  ],
  "Categories": [
      "Question Generation"
  ],
  "Reasoning": [],
  "Definition": [
      "In this task, you're given passages that contain mentions of names of people, places, or things. Some of these mentions refer to the same person, place, or thing. Your job is to write questions that evaluate one's understanding of such references. Good questions are expected to link pronouns (she, her, him, his, their, etc.) or other mentions to people, places, or things to which they may refer. Do not ask questions that can be answered correctly without understanding the paragraph or having multiple answers. Avoid questions that do not link phrases referring to the same entity. For each of your questions, the answer should be one or more phrases in the paragraph, and it should be unambiguous."
  ],
  "Input_language": [
      "English"
  ],
  "Output_language": [
      "English"
  ],
  "Instruction_language": [
      "English"
  ],
  "Domains": [
      "Wikipedia"
  ],
  "Positive Examples": [
      {
          "input": "Passage: Nearing London, Oliver encounters Jack Dawkins, a pickpocket more commonly known by the nickname the \"Artful Dodger\", and his sidekick, a boy of a humorous nature named Charley Bates, but Oliver's innocent and trusting nature fails to see any dishonesty in their actions. The Dodger provides Oliver with a free meal and tells him of a gentleman in London who will \"give him lodgings for nothing, and never ask for change\". Grateful for the unexpected assistance, Oliver follows the Dodger to the \"old gentleman's\" residence. In this way Oliver unwittingly falls in with an infamous Jewish criminal known as Fagin, the gentleman of whom the Artful Dodger spoke. Ensnared, Oliver lives with Fagin and his gang of juvenile pickpockets in their lair at Saffron Hill for some time, unaware of their criminal occupations. He believes they make wallets and handkerchiefs.",
          "output": "Who believes Fagin's gang make wallets and handkerchiefs?.",
          "explanation": "This question is based on the following sentence in the passage \"He believes they make wallets and handkerchiefs\". It evaluates the understanding that the pronoun \"he\" refers to name \"Oliver\". You can ask questions like this one about most pronouns in a paragraph."
      },
      ...
  ],
  "Negative Examples": [
      {
          "input": "Passage: In The Tangled Skein, Queen Mary is characterized as a loving woman with a strong sense of justice. The tangled skein arises from Mary's love for the fictional character Robert d'Esclade, fifth Duke of Wessex, said in this book to be the people's choice as King Consort. Wessex is chivalrous and charming, but semi-betrothed to Lady Ursula Glynde, whom he has not seen since her infancy. Wessex is repelled by the idea of having his wife thrust upon him and purposely avoids Lady Ursula. Unknown to Wessex, the Queen jealously guards him against Ursula, who is extremely beautiful. As soon as she realizes the Queen is keeping her away from Wessex, Ursula is angered. She believes she loves Wessex, for his nobility and goodness, and she is invested heavily in the betrothal. On her father's deathbed, Ursula promised to go into a convent if she did not marry Wessex. Although Ursula does not want to lose her independence by marrying, she seeks to frustrate the Queen's plans and make Wessex notice her; however, the arrival of Cardinal de Moreno, and his henchman Don Mignel, Marquis de Saurez, shifts the scene.",
          "output": "Who does Ursula promise to?.",
          "explanation": "There is only one mention of Ursula's (\"her\") father in the sentence \"On her father's deathbed, Ursula promised to go into a convent if she did not marry Wessex\". Hence, this question does not require linking two different phrases that refer to the same entity. When you write questions, it will be helpful to think about the phrases referring to entities that they link. If you cannot think of more than one such phrase, it is likely a bad question."
      },
      ...
  ],
  "Instances": [
        {
            "id": "task001-f44801d948324957abe71877d837d070",
            "input": "Passage: The earthquake swarm was noted on October 12, 2007 in the Prince George Citizen by citizen staff, three days after the earthquakes began. Scientists mentioned in the report were seismologist John Cassidy of Natural Resources Canada and volcanologist Catherine Hickson, who was part of the Geological Survey of Canada at the time. At the time of the report, scientists did not know the origin of the swarm. Seismologist John Cassidy stated, \"the depth is enough to rule out hydrothermal but it's up in the air as to whether the cause is tectonic shifts or volcanic activity. If it is volcanic there are certain characteristics that we would expect, there's a tremor-like character to it. And so we'll be looking for the types of events that we see beneath volcanoes and we'll be looking to see if they're getting closer to the surface or if they're migrating at all.\"Even if the Nazko swarm were a warning of a volcanic eruption, Hickson doubted it would turn out to be a highly explosive eruption like those that can occur in subduction-zone volcanoes. \"We're not talking about an injection of tonnes of ash many kilometers into the air like the 1980 Mount St. Helens eruption or the 1991 Mount Pinatubo eruption. We're talking about something very small, relatively localized that should have a fairly limited impact... but it'll be extremely exciting\", Hickson said. If an eruption were to occur, Hickson suggested that it would be characterized by a lava fountain that sends globs of lava 100 m (330 ft) into the air. This is similar to those that occur in Hawaii. Hickson said that a Nazko eruption could be a tourist attraction, but warned that noxious gases such as carbon dioxide and sulfur dioxide would be released during the event.",
            "output": [
                "What is the first name of the person who doubted it would turn out to be a highly explosive eruption like those that can occur in subduction-zone volcanoes?"
            ]
        },
        ...
    ],
  "Instance License": [
      "CC BY 4.0"
  ]
}
```

## Integrating Alpaca

We refer [Exploring Format Consistency for Instruction Tuning](https://arxiv.org/abs/2307.15504) and use heuristic method introduced in this work to integrate Alpaca into Super-NaturalInstructions.

## Training Examples

The prompt that tk-instruct use in definition-only is:

```txt
Definition : {{definition}}

Now complete the following example−
input : {{x.input}}
output :
```

and an example from alpaca is:
```json
{
    "instruction": "Give three tips for staying healthy.",
    "input": "",
    "output": "1.Eat a balanced diet and make sure to include plenty of fruits and vegetables. \n2. Exercise regularly to keep your body active and strong. \n3. Get enough sleep and maintain a consistent sleep schedule."
}
```

so the training examples from alpaca would look like:
```txt
Definition : Give three tips for staying healthy.

Now complete the following example−
input :
output :
```

and the target is:
```txt
1.Eat a balanced diet and make sure to include plenty of fruits and vegetables. \n2. Exercise regularly to keep your body active and strong. \n3. Get enough sleep and maintain a consistent sleep schedule.
```

## Main Results

The results are as follows:

| Dataset                       | EM    | ROUGE-L |
|-------------------------------|-------|---------|
| super(paper)                  | ?     | 45      |
| super(reproduce)              | 28.25 | 46.21   |
| super+raw(alpaca)             | 27.65 | 45.89   |
| super+heuristic(alpaca)       | 28.67 | 47.9    |
| super+heuristic(promptsource) | 33.02 | 49.78   |

### Analysis
1. 



- This repo is based on the released Tk-Instruct model in the [Super-NaturalInstructions paper](https://arxiv.org/abs/2204.07705).
- Tk-Instruct is a preliminary attempt towards general-purpose AI that can solve many NLP tasks by following in-context instructions (plain language task definitions or k-shot examples).
- It is built based on the pretrained [T5 model](https://arxiv.org/abs/1910.10683), and finetuned on our [data](https://github.com/allenai/natural-instructions).
- You can play with the 11B model via our online [demo](https://instructions.apps.allenai.org/demo)!

## Requirements

Our main experiments and analysis are conducted on the following environment:

- CUDA (11.3)
- cuDNN (8.2.0.53)
- Pytorch (1.10.0)
- Transformers (4.17.0)
- DeepSpeed

You can refer to the [Dockerfile](Dockerfile) for setting up the environment and install the required python libraries by running

```bash
pip install -r requirements.txt
```

Note: after the main exploration with 3B model, we train our 11B model on TPUs using the T5 code [here](https://github.com/google-research/text-to-text-transfer-transformer).

## Data

Our models are trained and evaluated on [Super-NaturalInstructions](https://github.com/allenai/natural-instructions), which can be cloned by running:

```bash
git clone git@github.com:allenai/natural-instructions.git data
```

Since Super-NaturalInstructions didn't provide an official split for the development set, in order to do evaluation during training time, you can mannualy create a `dev_tasks.txt` in the `data/splits/default` folder. We found it unclear what should be a meaningful validation set, under such cross-task generalization setting. You can use a part of the training tasks for validation, or you can set apart tasks in some categories for validation.

If you want to use the T5 code [here](https://github.com/google-research/text-to-text-transfer-transformer), you can convert the data into text2text format with [`scripts/convert_data_to_s2s.sh`](scripts/convert_data_to_s2s.sh).

## Training

A sample script for training the Tk-Instruct 3B model in our paper can be found at [`scripts/train_tk_instruct.sh`](scripts/train_tk_instruct.sh). You can run it as follows:

```bash
./scripts/train_tk_instruct.sh
```

However, if you are familiar with [Beaker](https://beaker.org/), you can refer to the [`beaker_configs/default_experiment.yaml`](beaker_configs/default_experiment.yaml) for a sample experiment config, and modifying [`src/create_exps.py`](src/create_exps.py) to easily starts a set of experiments by running:

```bash
python src/create_exps.py
```

## Released Checkpoints

Our 3B and 11B model checkpoints are accessible via the [Hugging Face Hub](https://huggingface.co/models?search=tk-instruct-). You can load them easily using the [Transformers](https://github.com/huggingface/transformers) library:

```python
>>> from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

>>> tokenizer = AutoTokenizer.from_pretrained("allenai/tk-instruct-3b-def")
>>> model = AutoModelForSeq2SeqLM.from_pretrained("allenai/tk-instruct-3b-def")

>>> input_ids = tokenizer.encode(
        "Definition: return the currency of the given country. Now complete the following example - Input: India. Output:", 
        return_tensors="pt"
    )
>>> output = model.generate(input_ids, max_length=10)
>>> output = tokenizer.decode(output[0], skip_special_tokens=True)
```

The model should generate `'Indian Rupee'` as the output.

## Evaluation

The following script evaluates our 3B Tk-Instruct model that uses `task definition + 2 positive examples` as instructions:

```bash
./scripts/eval_tk_instruct.sh
```

This should give you a ROUGE-L score of ~54.0, as is reported in the Table 3 of our [paper](https://arxiv.org/pdf/2204.07705.pdf).

You can also try other models under different encodings. You can control whether to include definition / explanation, or the number of pos/neg examples, by specifying the arguments in [`src/run_s2s.py`](src/run_s2s.py).

The numbers for heuristic baselines and GPT3 can be reproduced by using the following scripts:

```bash
./scripts/run_heuristics.sh
./scripts/run_gpt3.sh
```

## Model Predictions and Performance

The predictions of our tested models can be found in the [`output`](output/) folder. You can evaluate each predition file in the following way:

```bash
python src/compute_metrics.py --predictions output/default/tk-instruct-3b-def-pos/predicted_examples.jsonl --track default --compute_per_category_metrics
python src/compute_metrics.py --predictions output/xlingual/mtk-instruct-3b-def-pos/predicted_examples.jsonl --track xlingual --compute_per_category_metrics
```

Here are the performance numbers (in ROUGE-L) for our tested models:

|                          | Models                  | Default Track (en) | X-lingual Track |
|--------------------------|-------------------------|--------------------|-----------------|
| Heuristic Baselines      | Copying Instance Input  | 14.20              | 5.44            |
|                          | Copying Demo. Output    | 28.54              | 50.31           |
| Pretrained LMs           | T5-LM (11B)             | 30.16              | -               |
|                          | GPT3 (175B)             | 45.05              | 51.20           |
| Instruction-tuned Models | T0 (11B)                | 32.28              | -               |
|                          | GPT3-Instruct (175B)    | 52.06              | 53.74           |
|                          | Tk-Instruct (Ours, 3B)  | 54.33              | -               |
|                          | Tk-Instruct (Ours, 11B) | 60.07              | -               |
|                          | mTk-Instruct (Ours, 3B) | -                  | 56.72           |

Note that these numbers might be different from the numbers reported in the our arxiv paper, because we 1) resampled our evaluation instances; 2) updated our evaluation script. We will update the paper once allowed.

We will keep adding the predictions and performance of new models into this repository.

## Citation

```bib
@inproceedings{supernaturalinstructions,
  title={Super-NaturalInstructions:Generalization via Declarative Instructions on 1600+ Tasks},
  author={Wang, Yizhong and Mishra, Swaroop and Alipoormolabashi, Pegah and Kordi, Yeganeh and Mirzaei, Amirreza and Arunkumar, Anjana and Ashok, Arjun and Dhanasekaran, Arut Selvan and Naik, Atharva and Stap, David and others},
  booktitle={EMNLP},
  year={2022}
}
```
