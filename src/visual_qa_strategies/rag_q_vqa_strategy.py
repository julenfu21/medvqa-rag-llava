from pathlib import Path

from PIL import Image
from langchain_community.vectorstores import FAISS
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.retrievers import BaseRetriever
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama

from utils.enums import VQAStrategyType
from visual_qa_strategies.base_vqa_strategy import BaseVQAStrategy


class RagQVQAStrategy(BaseVQAStrategy):

    def _init_strategy(
        self,
        index_dir: Path,
        index_name: str,
        embedding_model_name: str,
        relevant_docs_count: int
    ) -> None:
        self.__retriever = self.__load_wikimed_retriever(
            index_dir,
            index_name,
            embedding_model_name,
            relevant_docs_count
        )


    @property
    def strategy_type(self) -> VQAStrategyType:
        return VQAStrategyType.RAG_Q


    def __load_wikimed_retriever(
        self,
        index_dir: Path,
        index_name: str,
        embedding_model_name: str,
        relevant_docs_count: int
    ) -> BaseRetriever:
        # Load embeddings
        print("\t- Loading Embeddings ...")
        embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model_name,
            # model_kwargs={'device': 'cuda'},
            encode_kwargs={'normalize_embeddings': False}
        )
        print("\t+ Embeddings Loaded.")

        # Load FAISS index
        print("\t- Loading Index ...")
        index = FAISS.load_local(
            folder_path=index_dir,
            index_name=index_name,
            embeddings=embeddings,
            allow_dangerous_deserialization=True
        )
        print("\t+ Index Loaded.")

        # Load retriever from index
        print("\t- Loading Retriever ...")
        retriever = index.as_retriever(
            search_type="similarity",
            search_kwargs={"k": relevant_docs_count}
        )
        print("\t+ Retriever Loaded.")
        return retriever


    def load_ollama_model(self, model_name: str) -> BaseChatModel:

        def prompt_template(data: dict) -> list:
            question = data["question"]
            image = data["image"]
            relevant_docs = data["relevant_docs"]

            return [
                SystemMessage(
                    content=(
                        "You are an assistant that only responds with a single letter: A, B, C, or "
                        "D. For each question, you should consider the provided options and the"
                        "image, and answer with exactly one letter that best matches the correct "
                        "choice. Answer with a single letter only, without any explanations or "
                        "additional information."
                    )
                ),
                HumanMessage(
                    content=[
                        {
                            "type": "image_url",
                            "image_url": f"data:image/jpeg;base64,{image}",
                        },
                        {
                            "type": "text",
                            "text": question
                        },
                        {
                            "type": "text",
                            "text": relevant_docs
                        }
                    ]
                ),
            ]

        llm = ChatOllama(model=model_name, temperature=0, num_predict=1)
        chain = prompt_template | llm | StrOutputParser()
        return chain


    def generate_answer_from_row(
        self,
        model: BaseChatModel,
        question: str,
        possible_answers: dict[str, str],
        image: Image.Image
    ) -> str:

        def format_docs(docs) -> str:
            return "\n\n".join(doc.page_content for doc in docs)

        possible_answers = " ".join(
            [f"{letter} - {answer}" for letter, answer in possible_answers.items()]
        )
        question_with_possible_answers = f"{question} {possible_answers}"

        output = model.invoke({
            "question": question_with_possible_answers,
            "image": image,
            "relevant_docs": format_docs(self.__retriever.invoke(question))
        })
        return output.strip()
