(base) tom@nvidia:~/github/wronai/intent$ cp .env.complete.example .env
(base) tom@nvidia:~/github/wronai/intent$ venv
(venv) (base) tom@nvidia:~/github/wronai/intent$ make install
Installing production dependencies...
pip3 install -e .
Obtaining file:///home/tom/github/wronai/intent
  Installing build dependencies ... done
  Checking if build backend supports build_editable ... done
  Getting requirements to build editable ... done
  Preparing editable metadata (pyproject.toml) ... done
Collecting pydantic>=2.0 (from intentforge==0.1.0)
  Using cached pydantic-2.12.5-py3-none-any.whl.metadata (90 kB)
Collecting pydantic-settings>=2.0 (from intentforge==0.1.0)
  Using cached pydantic_settings-2.12.0-py3-none-any.whl.metadata (3.4 kB)
Collecting python-dotenv>=1.0 (from intentforge==0.1.0)
  Using cached python_dotenv-1.2.1-py3-none-any.whl.metadata (25 kB)
Collecting anthropic>=0.30.0 (from intentforge==0.1.0)
  Using cached anthropic-0.75.0-py3-none-any.whl.metadata (28 kB)
Collecting paho-mqtt>=2.0 (from intentforge==0.1.0)
  Using cached paho_mqtt-2.1.0-py3-none-any.whl.metadata (23 kB)
Collecting anyio<5,>=3.5.0 (from anthropic>=0.30.0->intentforge==0.1.0)
  Using cached anyio-4.12.0-py3-none-any.whl.metadata (4.3 kB)
Collecting distro<2,>=1.7.0 (from anthropic>=0.30.0->intentforge==0.1.0)
  Using cached distro-1.9.0-py3-none-any.whl.metadata (6.8 kB)
Collecting docstring-parser<1,>=0.15 (from anthropic>=0.30.0->intentforge==0.1.0)
  Using cached docstring_parser-0.17.0-py3-none-any.whl.metadata (3.5 kB)
Collecting httpx<1,>=0.25.0 (from anthropic>=0.30.0->intentforge==0.1.0)
  Using cached httpx-0.28.1-py3-none-any.whl.metadata (7.1 kB)
Collecting jiter<1,>=0.4.0 (from anthropic>=0.30.0->intentforge==0.1.0)
  Using cached jiter-0.12.0-cp313-cp313-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (5.2 kB)
Collecting sniffio (from anthropic>=0.30.0->intentforge==0.1.0)
  Using cached sniffio-1.3.1-py3-none-any.whl.metadata (3.9 kB)
Collecting typing-extensions<5,>=4.10 (from anthropic>=0.30.0->intentforge==0.1.0)
  Using cached typing_extensions-4.15.0-py3-none-any.whl.metadata (3.3 kB)
Collecting idna>=2.8 (from anyio<5,>=3.5.0->anthropic>=0.30.0->intentforge==0.1.0)
  Using cached idna-3.11-py3-none-any.whl.metadata (8.4 kB)
Collecting certifi (from httpx<1,>=0.25.0->anthropic>=0.30.0->intentforge==0.1.0)
  Using cached certifi-2025.11.12-py3-none-any.whl.metadata (2.5 kB)
Collecting httpcore==1.* (from httpx<1,>=0.25.0->anthropic>=0.30.0->intentforge==0.1.0)
  Using cached httpcore-1.0.9-py3-none-any.whl.metadata (21 kB)
Collecting h11>=0.16 (from httpcore==1.*->httpx<1,>=0.25.0->anthropic>=0.30.0->intentforge==0.1.0)
  Using cached h11-0.16.0-py3-none-any.whl.metadata (8.3 kB)
Collecting annotated-types>=0.6.0 (from pydantic>=2.0->intentforge==0.1.0)
  Using cached annotated_types-0.7.0-py3-none-any.whl.metadata (15 kB)
Collecting pydantic-core==2.41.5 (from pydantic>=2.0->intentforge==0.1.0)
  Using cached pydantic_core-2.41.5-cp313-cp313-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (7.3 kB)
Collecting typing-inspection>=0.4.2 (from pydantic>=2.0->intentforge==0.1.0)
  Using cached typing_inspection-0.4.2-py3-none-any.whl.metadata (2.6 kB)
Using cached anthropic-0.75.0-py3-none-any.whl (388 kB)
Using cached anyio-4.12.0-py3-none-any.whl (113 kB)
Using cached distro-1.9.0-py3-none-any.whl (20 kB)
Using cached docstring_parser-0.17.0-py3-none-any.whl (36 kB)
Using cached httpx-0.28.1-py3-none-any.whl (73 kB)
Using cached httpcore-1.0.9-py3-none-any.whl (78 kB)
Using cached jiter-0.12.0-cp313-cp313-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (361 kB)
Using cached pydantic-2.12.5-py3-none-any.whl (463 kB)
Using cached pydantic_core-2.41.5-cp313-cp313-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (2.1 MB)
Using cached typing_extensions-4.15.0-py3-none-any.whl (44 kB)
Using cached annotated_types-0.7.0-py3-none-any.whl (13 kB)
Using cached h11-0.16.0-py3-none-any.whl (37 kB)
Using cached idna-3.11-py3-none-any.whl (71 kB)
Using cached paho_mqtt-2.1.0-py3-none-any.whl (67 kB)
Using cached pydantic_settings-2.12.0-py3-none-any.whl (51 kB)
Using cached python_dotenv-1.2.1-py3-none-any.whl (21 kB)
Using cached typing_inspection-0.4.2-py3-none-any.whl (14 kB)
Using cached certifi-2025.11.12-py3-none-any.whl (159 kB)
Using cached sniffio-1.3.1-py3-none-any.whl (10 kB)
Building wheels for collected packages: intentforge
  Building editable for intentforge (pyproject.toml) ... done
  Created wheel for intentforge: filename=intentforge-0.1.0-0.editable-py3-none-any.whl size=12127 sha256=442494319699e6a5ca5e44b498426bf43272a935d28afcb834e5e7876f40336b
  Stored in directory: /tmp/pip-ephem-wheel-cache-k1eflczu/wheels/e5/60/eb/e73552d2a4c78816af2fbb0c69ba39f904c3d95a64b901ee60
Successfully built intentforge
Installing collected packages: typing-extensions, sniffio, python-dotenv, paho-mqtt, jiter, idna, h11, docstring-parser, distro, certifi, annotated-types, typing-inspection, pydantic-core, httpcore, anyio, pydantic, httpx, pydantic-settings, anthropic, intentforge
Successfully installed annotated-types-0.7.0 anthropic-0.75.0 anyio-4.12.0 certifi-2025.11.12 distro-1.9.0 docstring-parser-0.17.0 h11-0.16.0 httpcore-1.0.9 httpx-0.28.1 idna-3.11 intentforge-0.1.0 jiter-0.12.0 paho-mqtt-2.1.0 pydantic-2.12.5 pydantic-core-2.41.5 pydantic-settings-2.12.0 python-dotenv-1.2.1 sniffio-1.3.1 typing-extensions-4.15.0 typing-inspection-0.4.2

[notice] A new release of pip is available: 25.1.1 -> 25.3
[notice] To update, run: pip install --upgrade pip
(venv) (base) tom@nvidia:~/github/wronai/intent$ make publish
Cleaning...
rm -rf build/ dist/ *.egg-info
rm -rf .pytest_cache .mypy_cache .ruff_cache
rm -rf htmlcov/ .coverage coverage.xml
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete
Building packages...
python3 -m build
/home/tom/github/wronai/intent/venv/bin/python3: No module named build
make: *** [Makefile:90: build] Error 1
(venv) (base) tom@nvidia:~/github/wronai/intent$ make dev
Installing development dependencies...
pip3 install -e ".[dev,test,docs]"
Obtaining file:///home/tom/github/wronai/intent
  Installing build dependencies ... done
  Checking if build backend supports build_editable ... done
  Getting requirements to build editable ... done
  Preparing editable metadata (pyproject.toml) ... done
Requirement already satisfied: pydantic>=2.0 in ./venv/lib/python3.13/site-packages (from intentforge==0.1.0) (2.12.5)
Requirement already satisfied: pydantic-settings>=2.0 in ./venv/lib/python3.13/site-packages (from intentforge==0.1.0) (2.12.0)
Requirement already satisfied: python-dotenv>=1.0 in ./venv/lib/python3.13/site-packages (from intentforge==0.1.0) (1.2.1)
Requirement already satisfied: anthropic>=0.30.0 in ./venv/lib/python3.13/site-packages (from intentforge==0.1.0) (0.75.0)
Requirement already satisfied: paho-mqtt>=2.0 in ./venv/lib/python3.13/site-packages (from intentforge==0.1.0) (2.1.0)
Collecting ruff>=0.1 (from intentforge==0.1.0)
  Using cached ruff-0.14.10-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (26 kB)
Collecting mypy>=1.8 (from intentforge==0.1.0)
  Using cached mypy-1.19.1-cp313-cp313-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl.metadata (2.2 kB)
Collecting pre-commit>=3.6 (from intentforge==0.1.0)
  Using cached pre_commit-4.5.1-py2.py3-none-any.whl.metadata (1.2 kB)
Collecting bandit>=1.7 (from intentforge==0.1.0)
  Downloading bandit-1.9.2-py3-none-any.whl.metadata (7.1 kB)
Collecting safety>=2.3 (from intentforge==0.1.0)
  Downloading safety-3.7.0-py3-none-any.whl.metadata (11 kB)
Collecting pytest>=8.0 (from intentforge==0.1.0)
  Using cached pytest-9.0.2-py3-none-any.whl.metadata (7.6 kB)
Collecting pytest-cov>=4.1 (from intentforge==0.1.0)
  Using cached pytest_cov-7.0.0-py3-none-any.whl.metadata (31 kB)
Collecting pytest-asyncio>=0.23 (from intentforge==0.1.0)
  Using cached pytest_asyncio-1.3.0-py3-none-any.whl.metadata (4.1 kB)
Collecting pytest-timeout>=2.2 (from intentforge==0.1.0)
  Using cached pytest_timeout-2.4.0-py3-none-any.whl.metadata (20 kB)
Collecting pytest-benchmark>=4.0 (from intentforge==0.1.0)
  Downloading pytest_benchmark-5.2.3-py3-none-any.whl.metadata (29 kB)
Requirement already satisfied: httpx>=0.27 in ./venv/lib/python3.13/site-packages (from intentforge==0.1.0) (0.28.1)
Collecting respx>=0.20 (from intentforge==0.1.0)
  Downloading respx-0.22.0-py2.py3-none-any.whl.metadata (4.1 kB)
Collecting mkdocs>=1.5 (from intentforge==0.1.0)
  Downloading mkdocs-1.6.1-py3-none-any.whl.metadata (6.0 kB)
Collecting mkdocs-material>=9.5 (from intentforge==0.1.0)
  Downloading mkdocs_material-9.7.1-py3-none-any.whl.metadata (19 kB)
Collecting mkdocstrings>=0.24 (from mkdocstrings[python]>=0.24; extra == "docs"->intentforge==0.1.0)
  Downloading mkdocstrings-1.0.0-py3-none-any.whl.metadata (16 kB)
Requirement already satisfied: anyio<5,>=3.5.0 in ./venv/lib/python3.13/site-packages (from anthropic>=0.30.0->intentforge==0.1.0) (4.12.0)
Requirement already satisfied: distro<2,>=1.7.0 in ./venv/lib/python3.13/site-packages (from anthropic>=0.30.0->intentforge==0.1.0) (1.9.0)
Requirement already satisfied: docstring-parser<1,>=0.15 in ./venv/lib/python3.13/site-packages (from anthropic>=0.30.0->intentforge==0.1.0) (0.17.0)
Requirement already satisfied: jiter<1,>=0.4.0 in ./venv/lib/python3.13/site-packages (from anthropic>=0.30.0->intentforge==0.1.0) (0.12.0)
Requirement already satisfied: sniffio in ./venv/lib/python3.13/site-packages (from anthropic>=0.30.0->intentforge==0.1.0) (1.3.1)
Requirement already satisfied: typing-extensions<5,>=4.10 in ./venv/lib/python3.13/site-packages (from anthropic>=0.30.0->intentforge==0.1.0) (4.15.0)
Requirement already satisfied: idna>=2.8 in ./venv/lib/python3.13/site-packages (from anyio<5,>=3.5.0->anthropic>=0.30.0->intentforge==0.1.0) (3.11)
Requirement already satisfied: certifi in ./venv/lib/python3.13/site-packages (from httpx>=0.27->intentforge==0.1.0) (2025.11.12)
Requirement already satisfied: httpcore==1.* in ./venv/lib/python3.13/site-packages (from httpx>=0.27->intentforge==0.1.0) (1.0.9)
Requirement already satisfied: h11>=0.16 in ./venv/lib/python3.13/site-packages (from httpcore==1.*->httpx>=0.27->intentforge==0.1.0) (0.16.0)
Requirement already satisfied: annotated-types>=0.6.0 in ./venv/lib/python3.13/site-packages (from pydantic>=2.0->intentforge==0.1.0) (0.7.0)
Requirement already satisfied: pydantic-core==2.41.5 in ./venv/lib/python3.13/site-packages (from pydantic>=2.0->intentforge==0.1.0) (2.41.5)
Requirement already satisfied: typing-inspection>=0.4.2 in ./venv/lib/python3.13/site-packages (from pydantic>=2.0->intentforge==0.1.0) (0.4.2)
Collecting PyYAML>=5.3.1 (from bandit>=1.7->intentforge==0.1.0)
  Using cached pyyaml-6.0.3-cp313-cp313-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl.metadata (2.4 kB)
Collecting stevedore>=1.20.0 (from bandit>=1.7->intentforge==0.1.0)
  Downloading stevedore-5.6.0-py3-none-any.whl.metadata (2.3 kB)
Collecting rich (from bandit>=1.7->intentforge==0.1.0)
  Using cached rich-14.2.0-py3-none-any.whl.metadata (18 kB)
Collecting click>=7.0 (from mkdocs>=1.5->intentforge==0.1.0)
  Using cached click-8.3.1-py3-none-any.whl.metadata (2.6 kB)
Collecting ghp-import>=1.0 (from mkdocs>=1.5->intentforge==0.1.0)
  Downloading ghp_import-2.1.0-py3-none-any.whl.metadata (7.2 kB)
Collecting jinja2>=2.11.1 (from mkdocs>=1.5->intentforge==0.1.0)
  Using cached jinja2-3.1.6-py3-none-any.whl.metadata (2.9 kB)
Collecting markdown>=3.3.6 (from mkdocs>=1.5->intentforge==0.1.0)
  Downloading markdown-3.10-py3-none-any.whl.metadata (5.1 kB)
Collecting markupsafe>=2.0.1 (from mkdocs>=1.5->intentforge==0.1.0)
  Using cached markupsafe-3.0.3-cp313-cp313-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl.metadata (2.7 kB)
Collecting mergedeep>=1.3.4 (from mkdocs>=1.5->intentforge==0.1.0)
  Downloading mergedeep-1.3.4-py3-none-any.whl.metadata (4.3 kB)
Collecting mkdocs-get-deps>=0.2.0 (from mkdocs>=1.5->intentforge==0.1.0)
  Downloading mkdocs_get_deps-0.2.0-py3-none-any.whl.metadata (4.0 kB)
Collecting packaging>=20.5 (from mkdocs>=1.5->intentforge==0.1.0)
  Using cached packaging-25.0-py3-none-any.whl.metadata (3.3 kB)
Collecting pathspec>=0.11.1 (from mkdocs>=1.5->intentforge==0.1.0)
  Using cached pathspec-0.12.1-py3-none-any.whl.metadata (21 kB)
Collecting pyyaml-env-tag>=0.1 (from mkdocs>=1.5->intentforge==0.1.0)
  Downloading pyyaml_env_tag-1.1-py3-none-any.whl.metadata (5.5 kB)
Collecting watchdog>=2.0 (from mkdocs>=1.5->intentforge==0.1.0)
  Using cached watchdog-6.0.0-py3-none-manylinux2014_x86_64.whl.metadata (44 kB)
Collecting python-dateutil>=2.8.1 (from ghp-import>=1.0->mkdocs>=1.5->intentforge==0.1.0)
  Using cached python_dateutil-2.9.0.post0-py2.py3-none-any.whl.metadata (8.4 kB)
Collecting platformdirs>=2.2.0 (from mkdocs-get-deps>=0.2.0->mkdocs>=1.5->intentforge==0.1.0)
  Using cached platformdirs-4.5.1-py3-none-any.whl.metadata (12 kB)
Collecting babel>=2.10 (from mkdocs-material>=9.5->intentforge==0.1.0)
  Using cached babel-2.17.0-py3-none-any.whl.metadata (2.0 kB)
Collecting backrefs>=5.7.post1 (from mkdocs-material>=9.5->intentforge==0.1.0)
  Downloading backrefs-6.1-py313-none-any.whl.metadata (3.0 kB)
Collecting colorama>=0.4 (from mkdocs-material>=9.5->intentforge==0.1.0)
  Using cached colorama-0.4.6-py2.py3-none-any.whl.metadata (17 kB)
Collecting mkdocs-material-extensions>=1.3 (from mkdocs-material>=9.5->intentforge==0.1.0)
  Downloading mkdocs_material_extensions-1.3.1-py3-none-any.whl.metadata (6.9 kB)
Collecting paginate>=0.5 (from mkdocs-material>=9.5->intentforge==0.1.0)
  Downloading paginate-0.5.7-py2.py3-none-any.whl.metadata (11 kB)
Collecting pygments>=2.16 (from mkdocs-material>=9.5->intentforge==0.1.0)
  Using cached pygments-2.19.2-py3-none-any.whl.metadata (2.5 kB)
Collecting pymdown-extensions>=10.2 (from mkdocs-material>=9.5->intentforge==0.1.0)
  Downloading pymdown_extensions-10.19.1-py3-none-any.whl.metadata (3.1 kB)
Collecting requests>=2.30 (from mkdocs-material>=9.5->intentforge==0.1.0)
  Using cached requests-2.32.5-py3-none-any.whl.metadata (4.9 kB)
Collecting mkdocs-autorefs>=1.4 (from mkdocstrings>=0.24->mkdocstrings[python]>=0.24; extra == "docs"->intentforge==0.1.0)
  Downloading mkdocs_autorefs-1.4.3-py3-none-any.whl.metadata (13 kB)
Collecting mkdocstrings-python>=1.16.2 (from mkdocstrings[python]>=0.24; extra == "docs"->intentforge==0.1.0)
  Downloading mkdocstrings_python-2.0.1-py3-none-any.whl.metadata (13 kB)
Collecting griffe>=1.13 (from mkdocstrings-python>=1.16.2->mkdocstrings[python]>=0.24; extra == "docs"->intentforge==0.1.0)
  Downloading griffe-1.15.0-py3-none-any.whl.metadata (5.2 kB)
Collecting mypy_extensions>=1.0.0 (from mypy>=1.8->intentforge==0.1.0)
  Using cached mypy_extensions-1.1.0-py3-none-any.whl.metadata (1.1 kB)
Collecting librt>=0.6.2 (from mypy>=1.8->intentforge==0.1.0)
  Using cached librt-0.7.5-cp313-cp313-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl.metadata (1.3 kB)
Collecting cfgv>=2.0.0 (from pre-commit>=3.6->intentforge==0.1.0)
  Using cached cfgv-3.5.0-py2.py3-none-any.whl.metadata (8.9 kB)
Collecting identify>=1.0.0 (from pre-commit>=3.6->intentforge==0.1.0)
  Using cached identify-2.6.15-py2.py3-none-any.whl.metadata (4.4 kB)
Collecting nodeenv>=0.11.1 (from pre-commit>=3.6->intentforge==0.1.0)
  Downloading nodeenv-1.10.0-py2.py3-none-any.whl.metadata (24 kB)
Collecting virtualenv>=20.10.0 (from pre-commit>=3.6->intentforge==0.1.0)
  Using cached virtualenv-20.35.4-py3-none-any.whl.metadata (4.6 kB)
Collecting iniconfig>=1.0.1 (from pytest>=8.0->intentforge==0.1.0)
  Using cached iniconfig-2.3.0-py3-none-any.whl.metadata (2.5 kB)
Collecting pluggy<2,>=1.5 (from pytest>=8.0->intentforge==0.1.0)
  Using cached pluggy-1.6.0-py3-none-any.whl.metadata (4.8 kB)
Collecting py-cpuinfo (from pytest-benchmark>=4.0->intentforge==0.1.0)
  Using cached py_cpuinfo-9.0.0-py3-none-any.whl.metadata (794 bytes)
Collecting coverage>=7.10.6 (from coverage[toml]>=7.10.6->pytest-cov>=4.1->intentforge==0.1.0)
  Using cached coverage-7.13.0-cp313-cp313-manylinux1_x86_64.manylinux_2_28_x86_64.manylinux_2_5_x86_64.whl.metadata (8.5 kB)
Collecting six>=1.5 (from python-dateutil>=2.8.1->ghp-import>=1.0->mkdocs>=1.5->intentforge==0.1.0)
  Using cached six-1.17.0-py2.py3-none-any.whl.metadata (1.7 kB)
Collecting charset_normalizer<4,>=2 (from requests>=2.30->mkdocs-material>=9.5->intentforge==0.1.0)
  Using cached charset_normalizer-3.4.4-cp313-cp313-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl.metadata (37 kB)
Collecting urllib3<3,>=1.21.1 (from requests>=2.30->mkdocs-material>=9.5->intentforge==0.1.0)
  Using cached urllib3-2.6.2-py3-none-any.whl.metadata (6.6 kB)
Collecting authlib>=1.2.0 (from safety>=2.3->intentforge==0.1.0)
  Downloading authlib-1.6.6-py2.py3-none-any.whl.metadata (9.8 kB)
Collecting dparse>=0.6.4 (from safety>=2.3->intentforge==0.1.0)
  Using cached dparse-0.6.4-py3-none-any.whl.metadata (5.5 kB)
Collecting filelock<4.0,>=3.16.1 (from safety>=2.3->intentforge==0.1.0)
  Using cached filelock-3.20.1-py3-none-any.whl.metadata (2.1 kB)
Collecting marshmallow>=3.15.0 (from safety>=2.3->intentforge==0.1.0)
  Downloading marshmallow-4.1.2-py3-none-any.whl.metadata (7.4 kB)
Collecting nltk>=3.9 (from safety>=2.3->intentforge==0.1.0)
  Using cached nltk-3.9.2-py3-none-any.whl.metadata (3.2 kB)
Collecting ruamel-yaml>=0.17.21 (from safety>=2.3->intentforge==0.1.0)
  Downloading ruamel_yaml-0.18.17-py3-none-any.whl.metadata (27 kB)
Collecting safety-schemas==0.0.16 (from safety>=2.3->intentforge==0.1.0)
  Downloading safety_schemas-0.0.16-py3-none-any.whl.metadata (1.1 kB)
Collecting tenacity>=8.1.0 (from safety>=2.3->intentforge==0.1.0)
  Using cached tenacity-9.1.2-py3-none-any.whl.metadata (1.2 kB)
Collecting tomlkit (from safety>=2.3->intentforge==0.1.0)
  Using cached tomlkit-0.13.3-py3-none-any.whl.metadata (2.8 kB)
Collecting typer>=0.16.0 (from safety>=2.3->intentforge==0.1.0)
  Downloading typer-0.21.0-py3-none-any.whl.metadata (16 kB)
Collecting cryptography (from authlib>=1.2.0->safety>=2.3->intentforge==0.1.0)
  Using cached cryptography-46.0.3-cp311-abi3-manylinux_2_34_x86_64.whl.metadata (5.7 kB)
Collecting joblib (from nltk>=3.9->safety>=2.3->intentforge==0.1.0)
  Downloading joblib-1.5.3-py3-none-any.whl.metadata (5.5 kB)
Collecting regex>=2021.8.3 (from nltk>=3.9->safety>=2.3->intentforge==0.1.0)
  Using cached regex-2025.11.3-cp313-cp313-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl.metadata (40 kB)
Collecting tqdm (from nltk>=3.9->safety>=2.3->intentforge==0.1.0)
  Using cached tqdm-4.67.1-py3-none-any.whl.metadata (57 kB)
Collecting ruamel.yaml.clib>=0.2.15 (from ruamel-yaml>=0.17.21->safety>=2.3->intentforge==0.1.0)
  Using cached ruamel_yaml_clib-0.2.15-cp313-cp313-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl.metadata (3.5 kB)
Collecting shellingham>=1.3.0 (from typer>=0.16.0->safety>=2.3->intentforge==0.1.0)
  Using cached shellingham-1.5.4-py2.py3-none-any.whl.metadata (3.5 kB)
Collecting markdown-it-py>=2.2.0 (from rich->bandit>=1.7->intentforge==0.1.0)
  Using cached markdown_it_py-4.0.0-py3-none-any.whl.metadata (7.3 kB)
Collecting mdurl~=0.1 (from markdown-it-py>=2.2.0->rich->bandit>=1.7->intentforge==0.1.0)
  Using cached mdurl-0.1.2-py3-none-any.whl.metadata (1.6 kB)
Collecting distlib<1,>=0.3.7 (from virtualenv>=20.10.0->pre-commit>=3.6->intentforge==0.1.0)
  Using cached distlib-0.4.0-py2.py3-none-any.whl.metadata (5.2 kB)
Collecting cffi>=2.0.0 (from cryptography->authlib>=1.2.0->safety>=2.3->intentforge==0.1.0)
  Using cached cffi-2.0.0-cp313-cp313-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (2.6 kB)
Collecting pycparser (from cffi>=2.0.0->cryptography->authlib>=1.2.0->safety>=2.3->intentforge==0.1.0)
  Using cached pycparser-2.23-py3-none-any.whl.metadata (993 bytes)
Downloading bandit-1.9.2-py3-none-any.whl (134 kB)
Downloading mkdocs-1.6.1-py3-none-any.whl (3.9 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 3.9/3.9 MB 153.1 kB/s eta 0:00:00
Using cached click-8.3.1-py3-none-any.whl (108 kB)
Downloading ghp_import-2.1.0-py3-none-any.whl (11 kB)
Using cached jinja2-3.1.6-py3-none-any.whl (134 kB)
Downloading markdown-3.10-py3-none-any.whl (107 kB)
Using cached markupsafe-3.0.3-cp313-cp313-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl (22 kB)
Downloading mergedeep-1.3.4-py3-none-any.whl (6.4 kB)
Downloading mkdocs_get_deps-0.2.0-py3-none-any.whl (9.5 kB)
Downloading mkdocs_material-9.7.1-py3-none-any.whl (9.3 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 9.3/9.3 MB 158.7 kB/s eta 0:00:00
Using cached babel-2.17.0-py3-none-any.whl (10.2 MB)
Downloading backrefs-6.1-py313-none-any.whl (400 kB)
Using cached colorama-0.4.6-py2.py3-none-any.whl (25 kB)
Downloading mkdocs_material_extensions-1.3.1-py3-none-any.whl (8.7 kB)
Downloading mkdocstrings-1.0.0-py3-none-any.whl (35 kB)
Downloading mkdocs_autorefs-1.4.3-py3-none-any.whl (25 kB)
Downloading mkdocstrings_python-2.0.1-py3-none-any.whl (105 kB)
Downloading griffe-1.15.0-py3-none-any.whl (150 kB)
Using cached mypy-1.19.1-cp313-cp313-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl (13.6 MB)
Using cached librt-0.7.5-cp313-cp313-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl (189 kB)
Using cached mypy_extensions-1.1.0-py3-none-any.whl (5.0 kB)
Using cached packaging-25.0-py3-none-any.whl (66 kB)
Downloading paginate-0.5.7-py2.py3-none-any.whl (13 kB)
Using cached pathspec-0.12.1-py3-none-any.whl (31 kB)
Using cached platformdirs-4.5.1-py3-none-any.whl (18 kB)
Using cached pre_commit-4.5.1-py2.py3-none-any.whl (226 kB)
Using cached cfgv-3.5.0-py2.py3-none-any.whl (7.4 kB)
Using cached identify-2.6.15-py2.py3-none-any.whl (99 kB)
Downloading nodeenv-1.10.0-py2.py3-none-any.whl (23 kB)
Using cached pygments-2.19.2-py3-none-any.whl (1.2 MB)
Downloading pymdown_extensions-10.19.1-py3-none-any.whl (266 kB)
Using cached pytest-9.0.2-py3-none-any.whl (374 kB)
Using cached pluggy-1.6.0-py3-none-any.whl (20 kB)
Using cached iniconfig-2.3.0-py3-none-any.whl (7.5 kB)
Using cached pytest_asyncio-1.3.0-py3-none-any.whl (15 kB)
Downloading pytest_benchmark-5.2.3-py3-none-any.whl (45 kB)
Using cached pytest_cov-7.0.0-py3-none-any.whl (22 kB)
Using cached coverage-7.13.0-cp313-cp313-manylinux1_x86_64.manylinux_2_28_x86_64.manylinux_2_5_x86_64.whl (252 kB)
Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)
Using cached python_dateutil-2.9.0.post0-py2.py3-none-any.whl (229 kB)
Using cached pyyaml-6.0.3-cp313-cp313-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl (801 kB)
Downloading pyyaml_env_tag-1.1-py3-none-any.whl (4.7 kB)
Using cached requests-2.32.5-py3-none-any.whl (64 kB)
Using cached charset_normalizer-3.4.4-cp313-cp313-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl (153 kB)
Using cached urllib3-2.6.2-py3-none-any.whl (131 kB)
Downloading respx-0.22.0-py2.py3-none-any.whl (25 kB)
Using cached ruff-0.14.10-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (14.2 MB)
Downloading safety-3.7.0-py3-none-any.whl (312 kB)
Downloading safety_schemas-0.0.16-py3-none-any.whl (39 kB)
Using cached filelock-3.20.1-py3-none-any.whl (16 kB)
Downloading authlib-1.6.6-py2.py3-none-any.whl (244 kB)
Using cached dparse-0.6.4-py3-none-any.whl (11 kB)
Downloading marshmallow-4.1.2-py3-none-any.whl (48 kB)
Using cached nltk-3.9.2-py3-none-any.whl (1.5 MB)
Using cached regex-2025.11.3-cp313-cp313-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl (803 kB)
Downloading ruamel_yaml-0.18.17-py3-none-any.whl (121 kB)
Using cached ruamel_yaml_clib-0.2.15-cp313-cp313-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl (782 kB)
Using cached six-1.17.0-py2.py3-none-any.whl (11 kB)
Downloading stevedore-5.6.0-py3-none-any.whl (54 kB)
Using cached tenacity-9.1.2-py3-none-any.whl (28 kB)
Downloading typer-0.21.0-py3-none-any.whl (47 kB)
Using cached rich-14.2.0-py3-none-any.whl (243 kB)
Using cached markdown_it_py-4.0.0-py3-none-any.whl (87 kB)
Using cached mdurl-0.1.2-py3-none-any.whl (10.0 kB)
Using cached shellingham-1.5.4-py2.py3-none-any.whl (9.8 kB)
Using cached virtualenv-20.35.4-py3-none-any.whl (6.0 MB)
Using cached distlib-0.4.0-py2.py3-none-any.whl (469 kB)
Using cached watchdog-6.0.0-py3-none-manylinux2014_x86_64.whl (79 kB)
Using cached cryptography-46.0.3-cp311-abi3-manylinux_2_34_x86_64.whl (4.5 MB)
Using cached cffi-2.0.0-cp313-cp313-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (219 kB)
Downloading joblib-1.5.3-py3-none-any.whl (309 kB)
Using cached py_cpuinfo-9.0.0-py3-none-any.whl (22 kB)
Using cached pycparser-2.23-py3-none-any.whl (118 kB)
Using cached tomlkit-0.13.3-py3-none-any.whl (38 kB)
Using cached tqdm-4.67.1-py3-none-any.whl (78 kB)
Building wheels for collected packages: intentforge
  Building editable for intentforge (pyproject.toml) ... done
  Created wheel for intentforge: filename=intentforge-0.1.0-0.editable-py3-none-any.whl size=12127 sha256=87cb33229e488b0ff8b1689680229ff3b1209793b1264b7d1529471f3ec60549
  Stored in directory: /tmp/pip-ephem-wheel-cache-ftqght65/wheels/e5/60/eb/e73552d2a4c78816af2fbb0c69ba39f904c3d95a64b901ee60
Successfully built intentforge
Installing collected packages: py-cpuinfo, paginate, distlib, watchdog, urllib3, tqdm, tomlkit, tenacity, stevedore, six, shellingham, ruff, ruamel.yaml.clib, regex, PyYAML, pygments, pycparser, pluggy, platformdirs, pathspec, packaging, nodeenv, mypy_extensions, mkdocs-material-extensions, mergedeep, mdurl, marshmallow, markupsafe, markdown, librt, joblib, iniconfig, identify, filelock, coverage, colorama, click, charset_normalizer, cfgv, backrefs, babel, virtualenv, ruamel-yaml, requests, pyyaml-env-tag, python-dateutil, pytest, pymdown-extensions, nltk, mypy, mkdocs-get-deps, markdown-it-py, jinja2, griffe, dparse, cffi, safety-schemas, rich, respx, pytest-timeout, pytest-cov, pytest-benchmark, pytest-asyncio, pre-commit, ghp-import, cryptography, typer, mkdocs, intentforge, bandit, authlib, safety, mkdocs-material, mkdocs-autorefs, mkdocstrings, mkdocstrings-python
  Attempting uninstall: intentforge
    Found existing installation: intentforge 0.1.0
    Uninstalling intentforge-0.1.0:
      Successfully uninstalled intentforge-0.1.0
Successfully installed PyYAML-6.0.3 authlib-1.6.6 babel-2.17.0 backrefs-6.1 bandit-1.9.2 cffi-2.0.0 cfgv-3.5.0 charset_normalizer-3.4.4 click-8.3.1 colorama-0.4.6 coverage-7.13.0 cryptography-46.0.3 distlib-0.4.0 dparse-0.6.4 filelock-3.20.1 ghp-import-2.1.0 griffe-1.15.0 identify-2.6.15 iniconfig-2.3.0 intentforge-0.1.0 jinja2-3.1.6 joblib-1.5.3 librt-0.7.5 markdown-3.10 markdown-it-py-4.0.0 markupsafe-3.0.3 marshmallow-4.1.2 mdurl-0.1.2 mergedeep-1.3.4 mkdocs-1.6.1 mkdocs-autorefs-1.4.3 mkdocs-get-deps-0.2.0 mkdocs-material-9.7.1 mkdocs-material-extensions-1.3.1 mkdocstrings-1.0.0 mkdocstrings-python-2.0.1 mypy-1.19.1 mypy_extensions-1.1.0 nltk-3.9.2 nodeenv-1.10.0 packaging-25.0 paginate-0.5.7 pathspec-0.12.1 platformdirs-4.5.1 pluggy-1.6.0 pre-commit-4.5.1 py-cpuinfo-9.0.0 pycparser-2.23 pygments-2.19.2 pymdown-extensions-10.19.1 pytest-9.0.2 pytest-asyncio-1.3.0 pytest-benchmark-5.2.3 pytest-cov-7.0.0 pytest-timeout-2.4.0 python-dateutil-2.9.0.post0 pyyaml-env-tag-1.1 regex-2025.11.3 requests-2.32.5 respx-0.22.0 rich-14.2.0 ruamel-yaml-0.18.17 ruamel.yaml.clib-0.2.15 ruff-0.14.10 safety-3.7.0 safety-schemas-0.0.16 shellingham-1.5.4 six-1.17.0 stevedore-5.6.0 tenacity-9.1.2 tomlkit-0.13.3 tqdm-4.67.1 typer-0.21.0 urllib3-2.6.2 virtualenv-20.35.4 watchdog-6.0.0

[notice] A new release of pip is available: 25.1.1 -> 25.3
[notice] To update, run: pip install --upgrade pip
pre-commit install
pre-commit installed at .git/hooks/pre-commit
(venv) (base) tom@nvidia:~/github/wronai/intent$ nmake test
Command 'nmake' not found, but there are 14 similar ones.
(venv) (base) tom@nvidia:~/github/wronai/intent$ make test
Running tests...
pytest tests/ -v --cov=intentforge --cov-report=html --cov-report=term-missing
=================================================================================================================== test session starts ====================================================================================================================
platform linux -- Python 3.13.5, pytest-9.0.2, pluggy-1.6.0
benchmark: 5.2.3 (defaults: timer=time.perf_counter disable_gc=False min_rounds=5 min_time=0.000005 max_time=1.0 calibration_precision=10 warmup=False warmup_iterations=100000)
rootdir: /home/tom/github/wronai/intent
configfile: pyproject.toml
plugins: cov-7.0.0, asyncio-1.3.0, respx-0.22.0, anyio-4.12.0, benchmark-5.2.3, timeout-2.4.0
asyncio: mode=Mode.AUTO, debug=False, asyncio_default_fixture_loop_scope=None, asyncio_default_test_loop_scope=function
collected 0 items                                                                                                                                                                                                                                          

================================================================================================================== no tests ran in 0.01s ===================================================================================================================
ERROR: file or directory not found: tests/
                                                                                                                                                                                                                                                            
make: *** [Makefile:67: test] Error 4
(venv) (base) tom@nvidia:~/github/wronai/intent$ 


przeanalizuj projekt i napisz co jest niepoprawnie zorganizowane w folderach, czy są jakieś duplikaty oraz co można zrobić, aby użycie tego rozwiazania było bezpieczniejsze i prostsze po stronie frontenduoraz działao z LLM ollama przez  liteLLM


popraw sposob pobierania zmiennych portow i hostow z .env .env.example
aby był jeden plik konfiguracyjny środowiska w docker, pyhton, itd, dodaj testy e2e




