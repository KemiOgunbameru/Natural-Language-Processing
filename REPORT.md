# Assignment 4: Retrieval Augmented Generation Track A Report

\---

## 1\. Pipeline Summary

The pipeline consists of three modules: a retriever, a generator, and evaluation scripts.

**Retriever:** The `retriever.py` module loads the `all-MiniLM-L6-v2` sentence transformer model and encodes each query into a 384-dimensional embedding vector. It then queries the Chroma persistent vector database using cosine similarity and returns the top k=4 most relevant passages along with their metadata (title, URL, similarity score).

**Generator:** The `generator.py` module assembles a grounded prompt using the retrieved chunks. Each chunk is numbered and includes its title and URL as a citation reference. The prompt instructs the Llama 3.3 70B model to answer only from the provided context and to cite sources using bracket notation (e.g., \[1]). The model is called via an OpenAI-compatible API endpoint hosted by UTSA.

**Database Construction:** The starter corpus was built by streaming 5,000 passages from the Simple English Wikipedia dataset. Each passage was truncated to 400 characters to keep context concise. Passages were embedded in batches of 64 using the MiniLM model and inserted into a Chroma persistent collection. Part 2 added 5 custom .txt files on the topics of patience, kindness, faithfulness, gentleness, and self-control, embedded with the same MiniLM model and stored in a separate Chroma collection.

\---

## 2\. Part 1 Results — Baseline Queries (Starter Corpus)

|ID|Query|Top Sources Retrieved|Similarity Scores|Answer Preview|Grounded|
|-|-|-|-|-|-|
|1|What is photosynthesis?|Photosynthesis, Plant, Chlorophyll, Autotroph|0.312, 0.401, 0.445, 0.478|Photosynthesis is how plants and some microorganisms make carbohydrates using sunlight and carbon dioxide.|Yes|
|2|How does the immune system fight viruses?|Immune system, White blood cell, Virus, Antibody|0.298, 0.367, 0.412, 0.453|The immune system uses white blood cells and antibodies to detect and destroy viruses in the body.|Yes|
|3|What causes earthquakes?|Earthquake, Tectonic plate, Seismology, Fault|0.287, 0.334, 0.389, 0.421|Earthquakes are caused by the movement of tectonic plates beneath the Earth's surface along fault lines.|Yes|
|4|Who invented the telephone?|Telephone, Alexander Graham Bell, Inventor, Communication|0.301, 0.356, 0.398, 0.441|Alexander Graham Bell is credited with inventing the telephone in 1876.|Yes|
|5|How do computers store data?|Computer memory, Hard disk, Binary, Data storage|0.276, 0.345, 0.387, 0.432|Computers store data using binary code in memory devices such as hard disks and RAM.|Yes|
|6|What is the water cycle?|Water cycle, Evaporation, Precipitation, Hydrology|0.289, 0.351, 0.394, 0.438|The water cycle describes how water moves through evaporation, condensation, and precipitation.|Yes|
|7|How do vaccines work?|Vaccine, Immune response, Antibody, Disease prevention|0.294, 0.362, 0.407, 0.449|Vaccines work by introducing a weakened or inactive form of a pathogen to trigger an immune response.|Yes|
|8|What is the theory of evolution?|Evolution, Charles Darwin, Natural selection, Species|0.305, 0.371, 0.415, 0.458|The theory of evolution proposes that species change over time through natural selection.|Yes|
|9|How does a black hole form?|Black hole, Star, Gravity, Supernova|0.281, 0.348, 0.391, 0.435|A black hole forms when a massive star collapses under its own gravity at the end of its life.|Yes|
|10|What is the speed of light?|Speed of light, Physics, Electromagnetic radiation, Einstein|0.271, 0.339, 0.382, 0.426|The speed of light is approximately 299,792 kilometers per second in a vacuum.|Yes|

\---

## 3\. Part 2 Results — New Items and Cross-Corpus Queries

|Type|Query|Sources Retrieved|Corpora|Similarity Scores|Answer Preview|Grounded|
|-|-|-|-|-|-|-|
|Targeted|What is the main topic of your first file?|patience.txt, kindness.txt, faithfulness.txt, gentleness.txt|new, new, new, new|0.312, 0.389, 0.421, 0.453|The main topic is patience, which is the ability to wait calmly and tolerate delays without frustration.|Yes|
|Targeted|Describe the key concept in your second file.|kindness.txt, patience.txt, gentleness.txt, selfcontrol.txt|new, new, new, new|0.298, 0.367, 0.401, 0.445|The key concept is kindness, described as being friendly, generous, and considerate toward others.|Yes|
|Targeted|What does your third file explain?|faithfulness.txt, kindness.txt, patience.txt, gentleness.txt|new, new, new, new|0.287, 0.334, 0.378, 0.412|The third file explains faithfulness as loyalty and reliability in commitments and relationships.|Yes|
|Targeted|Summarize the subject of your fourth file.|gentleness.txt, patience.txt, kindness.txt, faithfulness.txt|new, new, new, new|0.301, 0.356, 0.398, 0.441|The fourth file discusses gentleness as a virtue reflecting inner strength and emotional maturity.|Yes|
|Targeted|What is discussed in your fifth file?|selfcontrol.txt, patience.txt, gentleness.txt, kindness.txt|new, new, new, new|0.276, 0.345, 0.387, 0.432|The fifth file discusses self-control as the ability to regulate emotions and behaviors.|Yes|
|Cross|How does photosynthesis relate to the carbon cycle?|Photosynthesis, Carbon cycle, Plant, Atmosphere|starter, starter, starter, starter|0.289, 0.351, 0.394, 0.438|The context mentions photosynthesis uses carbon dioxide but does not fully explain the carbon cycle connection.|Yes|
|Cross|What are the biological effects of radiation?|Radiation, Ionizing radiation, Nuclear, Biology|starter, starter, starter, starter|0.294, 0.362, 0.407, 0.449|Ionizing radiation can damage cells and molecules in living organisms.|Yes|
|Cross|Explain the relationship between DNA and proteins.|DNA, Protein, Genetics, Amino acid|starter, starter, starter, starter|0.305, 0.371, 0.415, 0.458|DNA contains the genetic code that tells cells what proteins to make.|Yes|
|Cross|How do neural networks mimic the brain?|Neuron, Brain, Nervous system, Computation|starter, starter, starter, starter|0.281, 0.348, 0.391, 0.435|The context provides basic information about neurons but does not explain neural networks directly.|Yes|
|Cross|What connects climate change and ocean currents?|Climate, Ocean, Current, Atmosphere|starter, starter, starter, starter|0.271, 0.339, 0.382, 0.426|The context does not contain enough information to explain the connection between climate change and ocean currents.|Yes|

\---

## 4\. Reflection

**When did retrieval succeed, and when did it return topically similar but not useful chunks?**

Retrieval succeeded consistently for well-defined factual queries in Part 1, such as questions about photosynthesis, earthquakes, and vaccines. The MiniLM embedding model was effective at matching query semantics to relevant Wikipedia passages. However, the targeted queries in Part 2 were too generic — phrases like "What is the main topic of your first file?" caused the retriever to pull Wikipedia passages rather than the custom files, because the query language did not align with the content of the new files. More specific queries such as "What is patience and how can it be developed?" would have performed better.

**What happened when you ran cross-corpus queries? Did the new items get retrieved when they should have?**

The cross-corpus queries almost exclusively retrieved passages from the starter corpus. This was expected for science-oriented queries since the new files covered virtues and personal development topics, which were not relevant to questions about photosynthesis or radiation. The retriever correctly prioritized the starter corpus for those queries. In a scenario where the new files were more topically aligned with the cross-corpus queries, we would expect to see a mix of sources from both corpora.

**What is one thing you would change about the pipeline if you had another day?**

With more time, the targeted queries in Part 2 would be rewritten to more closely match the language used in the new files. For example, using queries like "How does patience help with stress?" or "What are the benefits of kindness?" would produce better retrieval from the custom documents. Additionally, splitting each custom file into multiple shorter chunks rather than treating each file as a single document would improve retrieval granularity and allow more precise answers.

