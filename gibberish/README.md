# RCP-Merging Gibberish Detector

This tool analyzes JSON files containing model outputs and uses the OpenAI API to determine if the text is "meaningful" or "gibberish" (e.g., mixed languages, garbled text, loops).

## Prerequisites

For other dataset apart from PubMedQA. Ensure you have Python installed and the OpenAI Python library:

```bash
pip install openai
chmod +x detect_gibberish.sh
bash ./detect_gibberish.sh
```

For PubMedQA Analysis:
```bash
chmod +x gibberish_detector_pubmed.sh
bash./gibberish_detector_pubmed.sh
```