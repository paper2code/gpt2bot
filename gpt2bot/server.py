#!/usr/bin/env python3

import configparser
import argparse
import logging
import random

from flask import Flask, jsonify, request

from model import download_model_folder, download_reverse_model_folder, load_model
from decoder import generate_response

app = Flask(__name__)

# Script arguments can include path of the config
arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--config', type=str, default="emlyon-chatbot.cfg")
arg_parser.add_argument('--host', type=str, default="0.0.0.0")
arg_parser.add_argument('--port', type=str, default="5011")
args = arg_parser.parse_args()

# Read the config
config = configparser.ConfigParser(allow_no_value=True)
with open(args.config) as f:
    config.read_file(f)

# Download and load main model
target_folder_name = download_model_folder(config)
model, tokenizer = load_model(target_folder_name, config)

# Download and load reverse model
use_mmi = config.getboolean('model', 'use_mmi')
if use_mmi:
    mmi_target_folder_name = download_reverse_model_folder(config)
    mmi_model, mmi_tokenizer = load_model(mmi_target_folder_name, config)
else:
    mmi_model = None
    mmi_tokenizer = None

@app.route('/query')
def query():
    # Parse parameters
    num_samples = config.getint('decoder', 'num_samples')
    max_turns_history = config.getint('decoder', 'max_turns_history')
    # app.logger.info("Running the chatbot...")
    turns = []
    question = request.args.get('question')
    # process question
    from_index = max(len(turns)-max_turns_history-1, 0) if max_turns_history >= 0 else 0

    # Generate bot messages
    bot_messages = generate_response(
        model, 
        tokenizer, 
        question, 
        config, 
        mmi_model=mmi_model, 
        mmi_tokenizer=mmi_tokenizer
    )
    if num_samples == 1:
        bot_message = bot_messages[0]
    else:
        # TODO: Select a message that is the most appropriate given the context
        # This way you can avoid loops
        bot_message = random.choice(bot_messages)
    app.logger.info('question: %s', question)
    app.logger.info('result >>> %s', bot_message)
    return jsonify(bot_message)

if __name__ == '__main__':
    handler = RotatingFileHandler('../logs/emlyon-gtp2reddit.log', maxBytes=10000, backupCount=1)
    handler.setLevel(logging.INFO)
    app.logger.addHandler(handler)
    app.run(host=host, port=port)
