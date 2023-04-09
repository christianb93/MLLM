# Mastering large language models

Mastering large language models - the GitHub repository for the LLM series at [leftasexercise.com](http://www.leftasexercise.com). Here you find Jupyter notebooks and scripts for the posts in my series on large language models. To run this locally, I recommend to create a virtual environment and to set up a Jupyter kernel spec pointing to that environment, like this.

```
git clone https://github.com/christianb93/MLLM
cd MLLM
python3 -m venv .venv
source .venv/bin/activate
pip3 install -r requirements.txt
pip3 install ipykernel
python3 -m ipykernel install --user --name=MLLM
```

This will create a virtual environment called *venv*, activate it, install the required dependencies and a Jupyter kernel. The last command sets up a kernel definition (stored in .local/share/jupyter/kernels) called MLLM. If you then start a Jupyter notebook server using

```
jupyter-lab
```

you will be able to select this environment. 

Note that the list of requirements might grow as the blog series proceeds, so you might want to go back from time rerun pip to make sure that you have libraries installed that we need.


