import streamlit as st
import os
import utils 

def reset_state():
    """
    Set all variables in the session state to their initial values
    """
    st.session_state['input_ids'] = []
    st.session_state['past_key_values'] = None
    st.session_state['history'] = ""
    st.session_state['bot'] = ""
    st.session_state['prompt'] = ""

def init_state():
    """
    Initialize state if not yet done
    """
    if "input_ids" not in st.session_state:
        reset_state()


@st.cache_resource(show_spinner = "Loading model...")
def load_model(model_id, device):
    """
    Load model
    """
    model, tokenizer = utils.get_model_and_tokenizer(model_id)
    model.to(device)
    #
    # Reset state. Note that the cache ist global across
    # sessions, so this is only called once. For another session created
    # by a different user, we also call init_state in the main flow
    # to make sure that these items are present in the session state
    # of this second session using the same model
    #
    reset_state()
    return model, tokenizer


def process_prompt():
    """
    Process a prompt
    """
    #
    # Get widget state to access current prompt
    #
    prompt = st.session_state.prompt
    if "" == prompt:
        return
    st.session_state.input_ids.extend(tokenizer.encode(prompt))
    st.session_state.input_ids.append(tokenizer.eos_token_id)
    generated, past_key_values, _ = utils.generate(model = model, 
                                            tokenizer = tokenizer, 
                                            input_ids = st.session_state.input_ids, 
                                            past_key_values = st.session_state.past_key_values, 
                                            temperature = st.session_state.temperature, debug = False)
    response = tokenizer.decode(generated).replace('<|endoftext|>', '')
    #
    # Prepare next turn
    #
    st.session_state.input_ids.extend(generated)
    #
    # Handle special case of empty history separately to avoid empty lines
    #
    if st.session_state.history != "":
        st.session_state.history = f"{st.session_state.history}\nUser: {prompt}\nBot:   {response}"
    else:
        st.session_state.history = f"User: {prompt}\nBot:   {response}"
    st.session_state.prompt = ""
    st.session_state.bot = response


#################################################
# Main
#################################################

# 
# Avoid tokenizer warning when switching device 
#
os.environ["TOKENIZERS_PARALLELISM"] = "false"


#
# Config
#
st.set_page_config(page_title = "DialoGPT")
#
# Select model and device
#
devices = ["CPU", "GPU"] if utils.cuda_available() else ['CPU']
model_ids = utils.models.keys()

with st.container():
    col1, col2 = st.columns([0.7, 0.3])
    model_id = col1.selectbox(label = "Select a model: ", 
                            options = model_ids, 
                            index = 1,
                            format_func = lambda model_id : f"{model_id} - {utils.get_model_name(model_id)}")
    device = col2.selectbox(label = "Select a device: ", options = devices)

#
# Select temperature
#
temperature = st.slider(label =  "Temperature",
                        min_value = .0,
                        max_value = 1.5,
                        value = 0.8,
                        key = "temperature")

#
# Load model
#
model, tokenizer = load_model(model_id, device.lower())

#
# Init state - this is done in load_model only for the first session
# that is attached to a model
#
init_state()

#
# Input
#
with st.container():
    st.text_input(label = "Enter your prompt: ", key = "prompt", on_change = process_prompt)
    col1, col2 = st.columns(2)
    col1.button(label = "Submit", on_click = process_prompt, use_container_width = True)
    col2.button(label = "Reset chat", on_click = reset_state, use_container_width = True)
#
# Display last bot reply
#
st.text_input(label = "Last reply of bot: ", value = "", disabled = True, key = "bot")
#
# Display history
#
st.text_area(label = "Full chat history", value = "", disabled = True, max_chars = None, key = "history", height = 200)


