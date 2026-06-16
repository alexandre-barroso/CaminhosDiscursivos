# Caminhos Discursivos / Discursive Paths / Caminos Discursivos

Audit-oriented source release for the analytical core of **Caminhos Discursivos**, a multilingual corpus-geometry method for estimating discursive paths between pairs of terms in Portuguese, English, and Spanish corpora.

Core reference file for citation and patent records:

```text
analysis/caminhos_pipeline.py
SHA-256: 3abc1071d231d340b0dca204c6bce48ec989ed1c44449e666fc30d1c44435ab7
```

## English

### What This Repository Contains

This repository contains the analytical source needed to inspect the process used by the Caminhos Discursivos web demonstration:

- `analysis/preprocess.py`: strict text normalization, tokenization, stopword filtering, and vocabulary capping for Portuguese, English, and Spanish.
- `analysis/build_vector_store.py`: conversion utility for normalized fastText `.vec` / `.vec.gz` files into compact SQLite vector stores.
- `analysis/caminhos_pipeline.py`: the main analytical pipeline: corpus preprocessing, vector lookup, corpus-specific vector fitting, PCA geometry, intermediate path estimation, localized text reports, and PDF/PNG figure generation.
- `fixtures/`: small public smoke-test corpora.
- `models/*.placeholder`: placeholders for the large pretrained vector stores used by the production runtime.

The repository is meant to support audit, citation, and methodological review of the analysis layer. It is not a deployable copy of the hosted web service.

### What Is Not Included

The hosted demonstration includes deployment-specific material that is outside the scope of this public source release: frontend assets, generated bundles, web styling, service wiring, queue/runtime wrappers, production configuration, installed dependency folders, transient workspaces, and the full pretrained vector stores.

Large vector stores are represented by placeholders because each production SQLite file is larger than 50 MB. Rebuild them from fastText vectors with `analysis/build_vector_store.py`, or place equivalent local files in `models/`:

- `models/cc.pt.300.sqlite`
- `models/cc.en.300.sqlite`
- `models/cc.es.300.sqlite`

### Method Summary

1. Each corpus is decoded and strictly cleaned: lowercase, diacritics removed, symbols and numbers removed, and language-specific stopwords filtered.
2. User terms are normalized with the same rules and must reduce to one alphabetic token each.
3. Corpus vocabulary is capped while preserving required query terms.
4. Vectors are loaded from a language-specific SQLite fastText store.
5. When the corpus is sufficiently informative, a corpus-local Word2Vec space is initialized from pretrained vectors and refined on the corpus.
6. PCA builds a reduced geometric space.
7. Candidate intermediate words are selected by persistence near the segment between the two query terms across several PCA dimensions.
8. Reports and figures are localized according to the selected UI language.

### Minimal Local Audit Run

Install Python dependencies:

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

Prepare local vector stores in `models/`, then create a manifest matching the pipeline schema and run:

```bash
python analysis/caminhos_pipeline.py --manifest manifest.json --output out
```

The output directory contains a localized text report, a JSON manifest, vector PDF figures, and PNG previews.

## Português

### O Que Este Repositório Contém

Este repositório contém o código analítico necessário para inspecionar o processo usado pela demonstração web de Caminhos Discursivos:

- `analysis/preprocess.py`: normalização estrita, tokenização, filtragem de stopwords e limite de vocabulário para português, inglês e espanhol.
- `analysis/build_vector_store.py`: utilitário que converte arquivos fastText `.vec` / `.vec.gz` normalizados em bases SQLite compactas.
- `analysis/caminhos_pipeline.py`: pipeline principal: pré-processamento dos corpora, consulta vetorial, ajuste de espaço vetorial por corpus, geometria via PCA, estimação de caminhos intermediários, relatórios localizados e geração de figuras PDF/PNG.
- `fixtures/`: pequenos corpora públicos para testes de fumaça.
- `models/*.placeholder`: marcadores para as grandes bases vetoriais usadas em produção.

O repositório serve para auditoria, citação e revisão metodológica da camada analítica. Ele não é uma cópia implantável do serviço web hospedado.

### O Que Não Está Incluído

A demonstração hospedada possui materiais específicos de implantação que estão fora do escopo desta publicação pública de código: ativos de frontend, pacotes gerados, estilos web, integração de serviço, wrappers de fila/execução, configuração de produção, pastas de dependências instaladas, áreas transitórias de trabalho e as bases vetoriais completas.

As bases vetoriais grandes são representadas por placeholders porque cada arquivo SQLite de produção ultrapassa 50 MB. Reconstrua-as a partir dos vetores fastText com `analysis/build_vector_store.py`, ou coloque arquivos locais equivalentes em `models/`:

- `models/cc.pt.300.sqlite`
- `models/cc.en.300.sqlite`
- `models/cc.es.300.sqlite`

### Resumo Do Método

1. Cada corpus é decodificado e limpo de forma estrita: minúsculas, diacríticos removidos, símbolos e números removidos, e stopwords filtradas por língua.
2. Os termos do usuário são normalizados pelas mesmas regras e precisam se reduzir a um único token alfabético cada.
3. O vocabulário do corpus é limitado preservando os termos consultados.
4. Vetores são carregados de uma base SQLite fastText específica da língua.
5. Quando o corpus tem informação suficiente, um espaço Word2Vec local é inicializado com vetores pré-treinados e refinado no corpus.
6. PCA constrói um espaço geométrico reduzido.
7. Palavras intermediárias candidatas são selecionadas por persistência perto do segmento entre os dois termos consultados em várias dimensões de PCA.
8. Relatórios e figuras são localizados conforme a língua da interface.

### Execução Local Mínima Para Auditoria

Instale as dependências Python:

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

Prepare as bases vetoriais locais em `models/`, crie um manifesto no schema esperado pelo pipeline e execute:

```bash
python analysis/caminhos_pipeline.py --manifest manifest.json --output out
```

A pasta de saída contém relatório textual localizado, manifesto JSON, figuras vetoriais em PDF e prévias PNG.

## Español

### Qué Contiene Este Repositorio

Este repositorio contiene el código analítico necesario para inspeccionar el proceso usado por la demostración web de Caminhos Discursivos:

- `analysis/preprocess.py`: normalización estricta, tokenización, filtrado de stopwords y límite de vocabulario para portugués, inglés y español.
- `analysis/build_vector_store.py`: utilidad para convertir archivos fastText `.vec` / `.vec.gz` normalizados en bases SQLite compactas.
- `analysis/caminhos_pipeline.py`: pipeline principal: preprocesamiento de corpus, consulta vectorial, ajuste de espacio vectorial por corpus, geometría con PCA, estimación de caminos intermedios, informes localizados y generación de figuras PDF/PNG.
- `fixtures/`: pequeños corpus públicos para pruebas rápidas.
- `models/*.placeholder`: marcadores para las grandes bases vectoriales usadas en producción.

El repositorio está pensado para auditoría, cita y revisión metodológica de la capa analítica. No es una copia desplegable del servicio web hospedado.

### Qué No Está Incluido

La demostración hospedada contiene materiales específicos de despliegue que quedan fuera del alcance de esta publicación pública de código: activos de frontend, paquetes generados, estilos web, integración de servicio, wrappers de cola/ejecución, configuración de producción, carpetas de dependencias instaladas, espacios de trabajo transitorios y las bases vectoriales completas.

Las bases vectoriales grandes se representan con placeholders porque cada archivo SQLite de producción supera los 50 MB. Reconstrúyalas a partir de vectores fastText con `analysis/build_vector_store.py`, o coloque archivos locales equivalentes en `models/`:

- `models/cc.pt.300.sqlite`
- `models/cc.en.300.sqlite`
- `models/cc.es.300.sqlite`

### Resumen Del Método

1. Cada corpus se decodifica y se limpia estrictamente: minúsculas, diacríticos eliminados, símbolos y números eliminados, y stopwords filtradas por lengua.
2. Los términos del usuario se normalizan con las mismas reglas y deben reducirse a un único token alfabético cada uno.
3. El vocabulario del corpus se limita preservando los términos consultados.
4. Los vectores se cargan desde una base SQLite fastText específica de la lengua.
5. Cuando el corpus tiene información suficiente, un espacio Word2Vec local se inicializa con vectores preentrenados y se refina en el corpus.
6. PCA construye un espacio geométrico reducido.
7. Las palabras intermedias candidatas se seleccionan por persistencia cerca del segmento entre los dos términos consultados en varias dimensiones de PCA.
8. Los informes y las figuras se localizan según la lengua de la interfaz.

### Ejecución Local Mínima Para Auditoría

Instale las dependencias Python:

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

Prepare las bases vectoriales locales en `models/`, cree un manifiesto con el schema esperado por el pipeline y ejecute:

```bash
python analysis/caminhos_pipeline.py --manifest manifest.json --output out
```

La carpeta de salida contiene un informe textual localizado, manifiesto JSON, figuras vectoriales en PDF y vistas PNG.

### Ref. (EN/PT/ES)

```bibtex
@misc{barroso2026caminhosdiscursivoswebapp,
  author       = {Barroso, A. M.},
  title        = {Caminhos Discursivos WebApp},
  year         = {2026},
  howpublished = {Programa de computador. Registro n. BR5120260042356},
  note         = {Registrado em 16 jun. 2026. Instituto Nacional da Propriedade Industrial (INPI)}
}

@misc{barroso2026discursivepathswebapp,
  author       = {Barroso, A. M.},
  title        = {Discursive Paths WebApp},
  year         = {2026},
  howpublished = {Computer program. Registration no. BR5120260042356},
  note         = {Registered on June 16, 2026. National Institute of Industrial Property (INPI)}
}

@misc{barroso2026caminosdiscursivoswebapp,
  author       = {Barroso, A. M.},
  title        = {Caminos Discursivos WebApp},
  year         = {2026},
  howpublished = {Programa de computadora. Registro n.º BR5120260042356},
  note         = {Registrado el 16 de junio de 2026. Instituto Nacional de la Propiedad Industrial (INPI)}
}
```

