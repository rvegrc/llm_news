# 📦 Стандартные библиотеки
import os
import re
from typing import List
from operator import itemgetter

# for .env file support and get access to api keys
from dotenv import load_dotenv
load_dotenv()

# 📦 Сторонние библиотеки
import pandas as pd
from pydantic import BaseModel, Field

# 🚀 LangChain и расширения
from langchain.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain_core.output_parsers import StrOutputParser
from langchain.output_parsers import (
    PydanticOutputParser,
    StructuredOutputParser,
    OutputFixingParser,
)
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.runnables import (
    Runnable,
    RunnablePassthrough,
    chain,
    RunnableConfig,
)

# 🔎 Graph Retriever
from langchain_graph_retriever import GraphRetriever
from graph_retriever.strategies import Eager
from langchain.schema import Document  # Если нужен именно этот Document
from dotenv import load_dotenv
import glob
from langchain_core.documents import Document
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_deepseek import ChatDeepSeek

from multiprocessing import Pool


# AllKeyMoments
class AspectOneKeyMoment(BaseModel):
    aspect_key_moment: str = Field(..., description="Выведи аспект ключевого момента")
    aspect_key_moment_comment: str = Field(..., description="Выведи подробный комментарий к аспекту ключевого момента")

class OneKeyMoment(BaseModel):
    # comment: str = Field(..., description="Выведи подробное описанием ключевого момента.")
    one_key_moment: str = Field(..., description='''
                            Выведи ключевой момент.
                        ''')
    all_aspects: List[AspectOneKeyMoment] = Field(..., description="Выведи все аспекты ключевого момента")
    one_key_moment_comment: str = Field(..., description="Выведи комментарий по ключевому моменту")

class AllKeyMoments(BaseModel):
    all_key_moments: List[OneKeyMoment] = Field(..., description="Выведи все ключевые моменты")
    all_key_moments_comment: str = Field(..., description="Выведи комментарий по всем ключевым моментам")



# OneKeyMomentRatio
class AspectRatio(BaseModel):
    aspect_id: int = Field(..., description="Идентификатор аспекта ключевого момента")
    aspect_ratio_comment: str = Field(..., description="""Сравни тестовый текст по аспекту ключевого момента.
                       Выведи комментарий к сравнению, что раскрыто полностью, частично или не раскрыто и почему ты так думаешь""")
    aspect_ratio: int = Field(..., description='''. 
                       На основании этого сравнения выведи полноту тестового текста 0 до 100, где 0 - аспект не раскрыт, 100 - полностью раскрыт''')
    
class OneKeyMomentRatio(BaseModel):
    all_aspects_ratio: List[AspectRatio] = Field(..., description="Выведи все аспекты ключевого момента с их полнотой раскрытия")
    one_key_moment_ratio_comment: str = Field(..., description="""Сравни тестовый текст по ключевому моменту.
                       Выведи комментарий к сравнению, что раскрыто полностью, частично или не раскрыто и почему ты так думаешь""")
    one_key_moment_ratio: int = Field(..., description='''На основании этого сравнения выведи полноту тестового текста по ключевому моменту от 0 до 100,
                       где 0 - ключевой момент не раскрыт, 100 - полностью раскрыт''')

llm = ChatDeepSeek(
    model="deepseek-reasoner",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

data_test_web = pd.read_csv("tmp/data_test_web.csv")[:2]


def get_all_key_moments(text_id):
    text = data_test_web.loc[text_id, 'web_text']

    parser = PydanticOutputParser(pydantic_object=AllKeyMoments)
    parser = OutputFixingParser.from_llm(parser=parser, llm=llm)

    prompt_key_points = ChatPromptTemplate.from_template(
    '''
    Выделить ключевые моменты текста {web_text} с подробным описанием каждого ключевого момента.
    Отметь подробно все важные аспекты каждого ключевого момента.
    Формируй вывод следующим образом:{template}
    '''
    )

    llm_chain = (
        prompt_key_points
        | llm 
        | parser

    )

    # web text analyze by key moments and aspects
    
    result = llm_chain.invoke({
        'template': parser.get_format_instructions(),
        "web_text": text        
    })
    
    return {'text_id': text_id, 'result': result}

def get_all_key_moments_ratio(text_id):
    row = results[results['text_id'] == text_id]
    wkm = row['result'].values[0]

    parser = PydanticOutputParser(pydantic_object=OneKeyMomentRatio)
    parser = OutputFixingParser.from_llm(parser=parser, llm=llm)

    prompt_ratio = ChatPromptTemplate.from_template(
        '''
        Проверь насколько полно в тестовом тексте: {lib_text}. раскрыт этот ключевой момент: {agg_key_moment}. Определи полноту раскрытия каждого аспекта в отдельности
        и полноту раскрытия ключевого момента как среднее значение по всем аспектам. Ответ выведи в следующем формате: {template}    
    '''
    )

    llm_chain_ratio = (
        prompt_ratio
        | llm
        | parser
    
    )

    df_key_moments = []

    for target_moment in wkm.all_key_moments:
        agg_key_moment = f'ключевой момент: {target_moment.one_key_moment}\n'
        agg_key_moment += f'комментарий к ключевому моменту: {target_moment.one_key_moment_comment}\n'
        agg_key_moment += 'аспекты ключевого момента:\n\n'
        for n, aspect in enumerate(target_moment.all_aspects):
            agg_key_moment += f'аспект c id={n}: {aspect.aspect_key_moment}\n'
            agg_key_moment += f'комментарий к аспекту c id={n}: {aspect.aspect_key_moment_comment}\n\n'


        result_ratio = llm_chain_ratio.invoke({
            'template': parser.get_format_instructions(),
            "lib_text": data_test_web["lib_text"].iloc[text_id],
            "agg_key_moment": agg_key_moment
        })

        aspects = []
        for aspect in result_ratio.all_aspects_ratio:
            aspects.append({
                    "aspect_id": aspect.aspect_id
                    ,"aspect_name": target_moment.all_aspects[aspect.aspect_id].aspect_key_moment
                    ,"aspect_comment": target_moment.all_aspects[aspect.aspect_id].aspect_key_moment_comment    
                    ,"aspect_ratio": aspect.aspect_ratio
                    ,"aspect_ratio_comment": aspect.aspect_ratio_comment
                })


        df_aspects = pd.DataFrame(aspects)
        df_aspects['key_moment'] = target_moment.one_key_moment
        df_aspects['key_moment_comment'] = target_moment.one_key_moment_comment
        df_aspects['key_moment_ratio'] = result_ratio.one_key_moment_ratio
        df_aspects['key_moment_ratio_comment'] = result_ratio.one_key_moment_ratio_comment

        df_key_moments.append(df_aspects)


        df_key_moments = pd.concat(df_key_moments, ignore_index=True).reset_index(drop=True)

        df_key_moments['text_id'] = text_id

        df_key_moments['text_ratio_mean'] = df_key_moments['key_moment_ratio'].mean()
        df_key_moments['text_ratio_median'] = df_key_moments['key_moment_ratio'].median()

    return df_key_moments


web_text_indexes = data_test_web.index.tolist()

if __name__ == "__main__":
    with Pool(processes=50) as pool:
        results = pd.DataFrame(pool.map(get_all_key_moments, web_text_indexes))

    with Pool(processes=50) as pool:
        all_key_moments_ratio = pd.DataFrame(pool.map(get_all_key_moments_ratio,web_text_indexes))
   
    df_output = pd.concat(all_key_moments_ratio, ignore_index=True).reset_index(drop=True)
    df_output.to_csv("output/web_key_moments_ratio.csv", index=False)

    data_test_web['text_id'] = data_test_web.index
    df_output_group_id = df_output.groupby('text_id').agg({
        'text_ratio_mean': 'mean',
        'text_ratio_median': 'median'
    }).reset_index()
    data_test_web = data_test_web.merge(df_output_group_id, on='text_id', how='left')

    data_test_web.to_csv("output/data_test_web.csv", index=False)