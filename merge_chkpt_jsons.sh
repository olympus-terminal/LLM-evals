#!/bin/bash

jq -s '.[0] * .[1] * .[2] * .[3] * .[4]' tokenizer.json tokenizer_config.json special_tokens_map.json adapter_config.json trainer_state.json > config.json
