THINGKING_PROMPT="""
You are a helpful assistant to answer the question by thinking step by step.
### INPUT ###
- Image: The image that serves as the basis for answering the question.
- Question: The question pertains to the content of the image.
- Answer: The correct answer for the question about the image.
### INSTRUCTION ###
- You should analyze the question and decide to focus on which visual content.
- You should parse the details of visual content based on the question.
- You should conclude the visual evidence to answer the question.
### OUTPUT ###
- The returned content MUST be in the natural flow.
<Image><Question><Answer>
"""

CONTRASTING_PROMPT="""
You are a helpful assistant to think step by step for discriminating between two images to answer two synonymous questions.
### INPUT ###
- First Image: One image that serves as the basis for answering the question.
- Second Image: The other image that serves as the basis for answering the question.
- First Question: The question pertains to the content of the First Image.
- Second Question: The question pertains to the content of the Second Image.
- Answer: The correct answer for the question about the images.
### INSTRUCTION ###
- When the correct answers for the two images are the same, you should summarize the common patterns in the visual content of the two images.
- When the correct answers for the two images are different, you should identify the differences in visual content between two images.
- Conclude the visual evidence to answer the questions respectively.
### OUTPUT ###
- Return in the natural flow.
<FirstImage><SecondImage>
<FirstQuestion><SecondQuestion><Answer>
"""

RETHINKING_PROMPT="""
You are a helpful assistant to rewrite the coarse rationale into a more correct and more logical one based on a contrastive analysis.
### INPUT ###
- Question: The question to be answered based one given target image.
- Answer: The correct answer to answer the question.
- Coarse Rationale: The naive reasoning process answering the question.
- Contrastive Analysis: The reasoning process when comparing the first image with the
second image for synonymous questions.
### INSTRUCTION ###
- The contrastive analysis is more reliable than the coarse rationale.
- If the answers in the contrastive analysis are the same for the two images, the model should formulate a summary reasoning schema. This schema must summarize the key visual features and confirm that the provided visual evidence aligns with this schema to derive the conclusion.
- If the answers in the contrastive analysis are different for the two images, you can employ backward chaining hypothesizing the visual cues that would be present if the alternative answer were correct, and then highlighting the critical distinctions between this hypothetical scenario and the actual visual evidence.
### OUTPUT ###
- The output MUST be in the format of '<think>the thinking content</think><answer>the answering content</answer>'.
- The content of thinking content MUST be between the special token of '<think>' and
'</think>'
- The content of answering content MUST be between the special token of '<answer>'
and '</answer>'.
<Question><Answer><CoarseRationale><ContrastiveAnalysis>
"""

FINAL_PROMPT="""
You are a helpful assistant to answer the question by thinking step by step.
### INPUT ###
- Image: The image that serves as the basis for answering the question.
- Question: The question pertains to the content of the image.
- Rationale: The rationale for the answer about the question.
### INSTRUCTION ###
- You should analyze the question and decide to focus on which visual content.
- You should parse the details of visual content based on the question.
- You should conclude the visual evidence to answer the question.
### OUTPUT ###
- The returned content MUST be in the natural flow.
<Image><Question><Rationale>
"""