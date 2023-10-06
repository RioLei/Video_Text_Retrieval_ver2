from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer_vi2en = AutoTokenizer.from_pretrained("vinai/vinai-translate-vi2en", src_lang="vi_VN")
model_vi2en = AutoModelForSeq2SeqLM.from_pretrained("vinai/vinai-translate-vi2en")

def translate_vi2en(vi_text: str) -> str:
    input_ids = tokenizer_vi2en(vi_text, return_tensors="pt").input_ids
    output_ids = model_vi2en.generate(
        input_ids,
        decoder_start_token_id=tokenizer_vi2en.lang_code_to_id["en_XX"],
        num_return_sequences=1,
        num_beams=5,
        early_stopping=True
    )
    en_text = tokenizer_vi2en.batch_decode(output_ids, skip_special_tokens=True)
    en_text = " ".join(en_text)
    return en_text

################ TESTING #################
# vi_text = "3 người đang đi bắt ếch trên một chiếc xuồng vào lúc chiều tối. Họ đều đeo đèn pin đội đầu. Có một người trong số đó mặc áo màu cam."
# print(translate_vi2en(vi_text))
