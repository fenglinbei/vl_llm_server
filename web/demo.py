# Copyright (c) Alibaba Cloud.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""A simple web interactive chat demo based on gradio."""

from argparse import ArgumentParser
from pathlib import Path

import copy
import gradio as gr
import os
import re
import sys
import cv2
import pprint
import secrets
import tempfile
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig

sys.path.insert(0, ".")
from api.utils.image_process import img_to_base64

from web.config import latex_delimiters_set, config
from api_test.api_test import API, draw_on
from api.utils.protocol import ChatMessage, Role, ContentType, Content, ChatCompletionResponse, Box

DEFAULT_CKPT_PATH = '/root/model/models--Qwen--Qwen-VL-Chat/snapshots/96a960aacb911cd09def34dc679bdb81f60e6110/'
BOX_TAG_PATTERN = r"<box>([\s\S]*?)</box>"
PUNCTUATION = "！？。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏."


def get_html(filename):
    path = os.path.join("./web/html/", filename)
    if os.path.exists(path):
        with open(path, encoding="utf8") as file:
            return file.read()
    return ""


def _get_args():
    parser = ArgumentParser()
    parser.add_argument("-c", "--checkpoint-path", type=str, default=DEFAULT_CKPT_PATH,
                        help="Checkpoint name or path, default to %(default)r")
    parser.add_argument("--cpu-only", action="store_true", help="Run demo with CPU only")

    parser.add_argument("--share", action="store_true", default=True,
                        help="Create a publicly shareable link for the interface.")
    parser.add_argument("--inbrowser", action="store_true", default=False,
                        help="Automatically launch the interface in a new tab on the default browser.")
    parser.add_argument("--server-port", type=int, default=8000,
                        help="Demo server port.")
    parser.add_argument("--server-name", type=str, default="192.168.1.210",
                        help="Demo server name.")

    args = parser.parse_args()
    return args


def _load_model_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(
        args.checkpoint_path, trust_remote_code=True, resume_download=True,
    )

    if args.cpu_only:
        device_map = "cpu"
    else:
        device_map = "cuda"

    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint_path,
        device_map=device_map,
        trust_remote_code=True,
        resume_download=True,
        fp16=True
    ).eval()
    model.generation_config = GenerationConfig.from_pretrained(
        args.checkpoint_path, trust_remote_code=True, resume_download=True,
    )

    return model, tokenizer


def _parse_text(text):
    lines = text.split("\n")
    lines = [line for line in lines if line != ""]
    count = 0
    for i, line in enumerate(lines):
        if "```" in line:
            count += 1
            items = line.split("`")
            if count % 2 == 1:
                lines[i] = f'<pre><code class="language-{items[-1]}">'
            else:
                lines[i] = f"<br></code></pre>"
        else:
            if i > 0:
                if count % 2 == 1:
                    line = line.replace("`", r"\`")
                    line = line.replace("<", "&lt;")
                    line = line.replace(">", "&gt;")
                    line = line.replace(" ", "&nbsp;")
                    line = line.replace("*", "&ast;")
                    line = line.replace("_", "&lowbar;")
                    line = line.replace("-", "&#45;")
                    line = line.replace(".", "&#46;")
                    line = line.replace("!", "&#33;")
                    line = line.replace("(", "&#40;")
                    line = line.replace(")", "&#41;")
                    line = line.replace("$", "&#36;")
                lines[i] = "<br>" + line
    text = "".join(lines)
    return text


def _launch_demo(args):
    uploaded_file_dir = os.environ.get("GRADIO_TEMP_DIR") or str(
        Path(tempfile.gettempdir()) / "gradio"
    )

    def predict(_chatbot, task_history, max_new_tokens, temperature, top_p, top_k):
        chat_query = _chatbot[-1][0]
        query = task_history[-1][0]
        # print("User: " + _parse_text(query))
        history_cp = copy.deepcopy(task_history)
        full_response = ""

        history_filter = []
        pic_idx = 1
        pre = ""
        messages = []
        for i, (q, a) in enumerate(history_cp):
            if isinstance(q, (tuple, list)):
                image_path = q[0]
                image = cv2.imread(image_path)
                base64_str = img_to_base64(img_array=image)
                messages.append(ChatMessage(role=Role.USER, content_type=ContentType.IMAGE, content=Content(base64_image=base64_str)))
                # print(q)
                q = f'Picture {pic_idx}: <img>{q[0]}</img>'
                pre += q + '\n'
                pic_idx += 1
            else:
                messages.append(ChatMessage(role=Role.USER, content_type=ContentType.TEXT, content=Content(text=q)))
                if a:
                    messages.append(ChatMessage(role=Role.ASSISTANT, content_type=ContentType.TEXT, content=Content(text=a))) 
                pre += q
                history_filter.append((pre, a))
                pre = ""
        history, message = history_filter[:-1], history_filter[-1][0]
        # pprint.pprint(message)
        # pprint.pprint(history)
        # pprint.pprint(messages)
        res = API.chat(messages=messages, max_tokens=max_new_tokens, temperature=temperature, top_p=top_p, top_k=top_k)
        res = ChatCompletionResponse(**res)
        response_msg = res.choices[0].message
        
        # response, history = model.chat(tokenizer, message, history=history)
        
        if response_msg.content_type == ContentType.BOX:
            boxes = response_msg.content.box
            response = response_msg.content.text
            image = draw_on(img=cv2.imread(image_path), boxes=boxes)
            # image = tokenizer.draw_bbox_on_latest_picture(response, history)
            # if image is not None:
            temp_dir = secrets.token_hex(20)
            temp_dir = Path(uploaded_file_dir) / temp_dir
            temp_dir.mkdir(exist_ok=True, parents=True)
            name = f"tmp{secrets.token_hex(5)}.jpg"
            filename = temp_dir / name
            cv2.imwrite(str(filename), image)
            _chatbot[-1] = (_parse_text(chat_query), (str(filename),))
            chat_response = response.replace("<ref>", "")
            chat_response = chat_response.replace(r"</ref>", "")
            chat_response = re.sub(BOX_TAG_PATTERN, "", chat_response)
            if chat_response != "":
                _chatbot.append((None, chat_response))
        else:
            response = response_msg.content.text
            _chatbot[-1] = (_parse_text(chat_query), response)
        full_response = _parse_text(response)

        task_history[-1] = (query, full_response)
        # print("Qwen-VL-Chat: " + _parse_text(full_response))
        return _chatbot

    def regenerate(_chatbot, task_history, max_new_tokens, temperature, top_p, top_k):
        if not task_history:
            return _chatbot
        item = task_history[-1]
        if item[1] is None:
            return _chatbot
        task_history[-1] = (item[0], None)
        chatbot_item = _chatbot.pop(-1)
        if chatbot_item[0] is None:
            _chatbot[-1] = (_chatbot[-1][0], None)
        else:
            _chatbot.append((chatbot_item[0], None))
        return predict(_chatbot, task_history, max_new_tokens, temperature, top_p, top_k)

    def add_text(history, task_history, text):
        # print(task_history)
        task_text = text
        if len(text) >= 2 and text[-1] in PUNCTUATION and text[-2] not in PUNCTUATION:
            task_text = text[:-1]
        history = history + [(_parse_text(text), None)]
        task_history = task_history + [(task_text, None)]
        return history, task_history, ""

    def add_file(history, task_history, file):
        history = history + [((file.name,), None)]
        task_history = task_history + [((file.name,), None)]
        return history, task_history

    def reset_user_input():
        return gr.update(value="")

    def reset_state(task_history):
        task_history.clear()
        return []

    with gr.Blocks() as demo:
#         gr.Markdown("""\
# <p align="center"><img src="https://modelscope.cn/api/v1/models/qwen/Qwen-7B-Chat/repo?
# Revision=master&FilePath=assets/logo.jpeg&View=true" style="height: 80px"/><p>""")
        gr.Markdown("""<center><font size=8>Qwen-VL-Chat Bot</center>""")
        gr.Markdown(
            """\
<center><font size=3>This WebUI is based on Qwen-VL-Chat, developed by Alibaba Cloud. \
(本WebUI基于Qwen-VL-Chat打造，实现聊天机器人功能。)</center>""")
#         gr.Markdown("""\
# <center><font size=4>Qwen-VL <a href="https://modelscope.cn/models/qwen/Qwen-VL/summary">🤖 </a> 
# | <a href="https://huggingface.co/Qwen/Qwen-VL">🤗</a>&nbsp ｜ 
# Qwen-VL-Chat <a href="https://modelscope.cn/models/qwen/Qwen-VL-Chat/summary">🤖 </a> | 
# <a href="https://huggingface.co/Qwen/Qwen-VL-Chat">🤗</a>&nbsp ｜ 
# &nbsp<a href="https://github.com/QwenLM/Qwen-VL">Github</a></center>""")
        with gr.Column():
            with gr.Row():
                chatbot = gr.Chatbot(
                    label="Chuanhu Chat",
                    elem_id="chuanhu-chatbot",
                    latex_delimiters=latex_delimiters_set,
                    height=700,
                    show_label=False,
                    avatar_images=[config.user_avatar, config.bot_avatar],
                    show_share_button=False,
                )
            with gr.Row(elem_id="chatbot-footer"):
                with gr.Box(elem_id="chatbot-input-box"):
                    with gr.Row(elem_id="chatbot-input-row"):
                        gr.HTML(get_html("chatbot_more.html").format(
                            single_turn_label="单轮对话",
                            websearch_label="在线搜索",
                            upload_file_label="上传文件",
                            uploaded_files_label="知识库文件",
                            uploaded_files_tip="在工具箱中管理知识库文件"
                        ))
                        with gr.Row(elem_id="chatbot-input-tb-row"):
                            with gr.Column(min_width=225, scale=12):
                                user_input = gr.Textbox(
                                    elem_id="user-input-tb",
                                    show_label=False,
                                    placeholder="在这里输入",
                                    elem_classes="no-container",
                                    max_lines=5,
                                    # container=False
                                )
                            with gr.Column(min_width=42, scale=1, elem_id="chatbot-ctrl-btns"):
                                submitBtn = gr.Button(
                                    value="", variant="primary", elem_id="submit-btn")
                                cancelBtn = gr.Button(
                                    value="", variant="secondary", visible=False, elem_id="cancel-btn")
                    # Note: Buttons below are set invisible in UI. But they are used in JS.
                    with gr.Row(elem_id="chatbot-buttons", visible=False):
                        with gr.Column(min_width=120, scale=1):
                            emptyBtn = gr.Button(
                                "🧹 新的对话", elem_id="empty-btn"
                            )
                        with gr.Column(min_width=120, scale=1):
                            retryBtn = gr.Button(
                                "🔄 重新生成", elem_id="gr-retry-btn")
                        with gr.Column(min_width=120, scale=1):
                            delFirstBtn = gr.Button("🗑️ 删除最旧对话")
                        with gr.Column(min_width=120, scale=1):
                            delLastBtn = gr.Button(
                                "🗑️ 删除最新对话", elem_id="gr-dellast-btn")
                        with gr.Row(visible=False) as like_dislike_area:
                            with gr.Column(min_width=20, scale=1):
                                likeBtn = gr.Button(
                                    "👍", elem_id="gr-like-btn")
                            with gr.Column(min_width=20, scale=1):
                                dislikeBtn = gr.Button(
                                    "👎", elem_id="gr-dislike-btn")

        # chatbot = gr.Chatbot(label='Qwen-VL-Chat', elem_classes="control-height", height=750)
        # query = gr.Textbox(lines=2, label='Input')
        task_history = gr.State([])

        with gr.Row():
            empty_btn = gr.Button("🧹 Clear History (清除历史)")
            submit_btn = gr.Button("🚀 Submit (发送)")
            regen_btn = gr.Button("🤔️ Regenerate (重试)")
            addfile_btn = gr.UploadButton("📁 Upload (上传文件)", file_types=["image"])
        with gr.Accordion("Setting"):
             max_new_tokens = gr.Number(label="max_new_tokens", value=1024)
             temperature = gr.Slider(minimum=0, maximum=1, value=0.8, step=0.01, label="temperature")
             top_p = gr.Slider(minimum=0, maximum=1, value=0.8, step=0.01, label="top_p")
             top_k = gr.Slider(minimum=1, maximum=50, value=5, step=1, label="top_k")

        gr.on(
                triggers=[submit_btn.click, user_input.submit],
                fn=add_text,
                inputs=[chatbot, task_history, user_input],
                outputs=[chatbot, task_history]
            ).then(
                reset_user_input,
                [],
                [user_input]
            ).then(
                predict,
                [chatbot, task_history, max_new_tokens, temperature, top_p, top_k],
                [chatbot],
                show_progress=True
            )
        # submit_btn.click(add_text, [chatbot, task_history, query], [chatbot, task_history]).then(
        #     predict, [chatbot, task_history], [chatbot], show_progress=True
        # )
        # submit_btn.click(reset_user_input, [], [query])
        empty_btn.click(reset_state, [task_history], [chatbot], show_progress=True)
        regen_btn.click(regenerate, [chatbot, task_history, max_new_tokens, temperature, top_p, top_k], [chatbot], show_progress=True)
        addfile_btn.upload(add_file, [chatbot, task_history, addfile_btn], [chatbot, task_history], show_progress=True)

#         gr.Markdown("""\
# <font size=2>Note: This demo is governed by the original license of Qwen-VL. \
# We strongly advise users not to knowingly generate or allow others to knowingly generate harmful content, \
# including hate speech, violence, pornography, deception, etc. \
# (注：本演示受Qwen-VL的许可协议限制。我们强烈建议，用户不应传播及不应允许他人传播以下内容，\
# 包括但不限于仇恨言论、暴力、色情、欺诈相关的有害信息。)""")

    demo.queue().launch(
        share=args.share,
        inbrowser=args.inbrowser,
        server_port=args.server_port,
        server_name=args.server_name,
    )


def main():
    args = _get_args()

    # model, tokenizer = _load_model_tokenizer(args)

    _launch_demo(args)


if __name__ == '__main__':
    main()