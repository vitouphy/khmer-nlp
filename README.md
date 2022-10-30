# KhmerNLP: NLP Tools for Khmer Language

This package aims to hold a collection of NLP functions to for processing Khmer language.

## Installation
```
pip install khmernlp
```

## Usage

```python
import khmernlp

# Tokenize and part-of-speech tagging
khmernlp.tokenize(["ប្រាសាទអង្គរវត្តឬប្រាសាទអង្គរតូចមានទីតាំងស្ថិតនៅភាគខាងជើងនៃក្រុងសៀមរាបនៃខេត្តសៀមរាប"])

# Output
[(
    ['ប្រាសាទ', 'អង្គរ', 'វត្ត', 'ឬ', 'ប្រាសាទ', 'អង្គរ', 'តូច', 'មាន', 'ទីតាំង', 'ស្ថិត', 'នៅ', 'ភាគ', 'ខាង', 'ជើង', 'នៃ', 'ក្រុង', 'សៀមរាប', 'នៃ', 'ខេត្ត', 'សៀមរាប'], 
    ['NN', 'PN', 'NN', 'IN', 'NN', 'PN', 'JJ', 'VB', 'NN', 'VB', 'IN', 'NN', 'NN', 'NN', 'IN', 'NN', 'PN', 'IN', 'NN', 'PN']
)]
```