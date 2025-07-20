from typing import List, Optional, Sequence


class Document:
    def __init__(self, url: str, title: Optional[str], excerpt: Optional[str],
                 main_image: Optional[str], favicon: Optional[str]) -> None:
        self.url = url
        self.title = title
        self.excerpt = excerpt
        self.main_image = main_image
        self.favicon = favicon

    def to_text(self) -> str:
        """Returns the textual content used for indexing/searching."""
        return " ".join(filter(None, [self.title, self.excerpt]))

    def __repr__(self) -> str:
        return (
            "\n📄 Document\n"
            f" ├─ 🌐 URL       : {self.url or '–'}\n"
            f" ├─ 🏷️  Title     : {self.title or '–'}\n"
            f" ├─ 📝 Excerpt   : {self.excerpt or '–'}\n"
            f" ├─ 🖼️  Main Image: {self.main_image or '–'}\n"
            f" └─ 🧷 Favicon   : {self.favicon or '–'}"
        )


class DocumentStore:
    def __init__(self) -> None:
        self.documents: List[Document] = []

    def add_document(self, doc: Document) -> None:
        self.documents.append(doc)

    def get_all(self) -> List[Document]:
        return self.documents

    def get_documents_by_index(self, indices: Sequence[int]) -> List[Document]:
        return [self.documents[i] for i in indices]

    def get_all_texts(self) -> List[str]:
        return [doc.to_text() for doc in self.documents]

    def get_by_ids(self, ids: Sequence[str]) -> List[Document]:
        id_set = set(ids)
        return [doc for doc in self.documents if doc.url in id_set]

    def __getitem__(self, idx: int) -> Document:
        return self.documents[idx]

    def __len__(self) -> int:
        return len(self.documents)

    def __iter__(self):
        return iter(self.documents)
