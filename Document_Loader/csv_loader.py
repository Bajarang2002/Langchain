from langchain_community.document_loaders import CSVLoader


loader = CSVLoader(file_path="Social_Network_Ads.csv")

docs = loader.load()

print(docs[1])
print(len(docs))