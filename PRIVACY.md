# Publication privacy and provenance

Project author: **Gilmour Jia**

The distribution is a sanitized derivative of a local research archive. Local operating-system account names and absolute home-directory paths are replaced with neutral workspace paths. Private chat excerpts used as research-consent identifiers are replaced consistently in code and saved manifests. Word creator and last-modified-by properties are set to Gilmour Jia, and revision identifiers and application/company metadata are cleaned. PNG metadata is removed without changing pixels.

The release is scanned for common credential formats, quoted credential assignments, private local paths, email addresses and credential-bearing URLs. Archive members of Word, NumPy and model-checkpoint containers are inspected without executing pickle objects or loading strategy code. A pattern scan is a bounded check, not a guarantee against every possible secret format.

`00_说明与核验/source_manifest.csv` uses anonymized source identifiers and hashes of the sanitized published files. Historic hashes stored inside experiment records describe the historic run and can differ from the published source. `release_manifest.csv` lists the actual publication contents and SHA256 values.

The research and metric definitions are retained. Privacy editing is not a new experiment or a correction of the historical model limitations. Full data and training environments remain external dependencies.

AI assistance is disclosed in the manuscript and README. Bibliographic authorship and retained third-party attribution are not reassigned to the project author.
