# protocol2spec


# Run instruction with new PDF
1. Add/Upload PDF in data/ dir
2. Create index: python -m scripts.build_index data/Protocol.pdf
3. Update data/variable_lexical.yml (in production env this should be updated programmatically based on the specs or a list of variables)
