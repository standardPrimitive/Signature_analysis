import telebot
from telebot.types import InputMediaPhoto
from telebot.types import InlineKeyboardMarkup, InlineKeyboardButton
import os
from model import extract_signatures_from_documents
from model import text_output_signatures
from model import visualize_best_signatures

TOKEN = '6470861520:AAGIJMPNcAz06r0kYUL0IDNYQh0zDRf5LJA'
bot = telebot.TeleBot(TOKEN)

@bot.message_handler(commands=['start'])
def start(message):
    markup = InlineKeyboardMarkup()
    markup.row_width = 2
    markup.add(InlineKeyboardButton("Загрузить подписи пользователя в базу данных", callback_data='upload_signatures'),
               InlineKeyboardButton("Идентифицировать подпись пользователя", callback_data='identify_signature'))
    bot.reply_to(message, "Выберите действие:", reply_markup=markup)
    

@bot.callback_query_handler(func=lambda call: True)
def query_handler(call):
    if call.data == 'upload_signatures':
        bot.answer_callback_query(call.id)
        bot.send_message(call.message.chat.id, "Пожалуйста, загрузите фото подписей.")
    elif call.data == 'identify_signature':
        bot.answer_callback_query(call.id)
        bot.send_message(call.message.chat.id, "Пожалуйста, загрузите фото документа с подписью и укажите фамилию сотрудника.")

@bot.message_handler(content_types=['photo'])
def handle_photo(message):
    try:
        file_id = message.photo[-1].file_id
        file_info = bot.get_file(file_id)
        downloaded_file = bot.download_file(file_info.file_path)

        check_user = message.caption

        with open("Docs/photo.jpg", 'wb') as new_file:
            new_file.write(downloaded_file)
    
    except Exception as e:
        bot.reply_to(message, "Произошла ошибка при загрузке фото. Пожалуйста, попробуйте еще раз.")

    try:
        res_extract = extract_signatures_from_documents(docs_path, predict_path)
        bot.reply_to(message, res_extract)
    except Exception as e:
        bot.reply_to(message, "Ошибка при распознавании документа")
    try:
        text_output = text_output_signatures(check_user, employee_signs_path+check_user, predicts_signs_path)
        bot.reply_to(message, text_output) 
    except Exception as e:
        bot.reply_to(message, "Ошибка при выводе информации по валидности подписи. Вероятно, вы забыли указать фамилию.")
    try:
        img_output = visualize_best_signatures(employee_signs_path + check_user, predicts_signs_path)
        media_group = []
        for img_path in img_output:
            with open(img_path, 'rb') as photo:
                media_group.append(InputMediaPhoto(photo.read()))  # Важно использовать read(), чтобы передать содержимое файла
        if media_group:
            bot.send_media_group(message.chat.id, media_group)
    except Exception as e:
        bot.reply_to(message, "Ошибка при выводе изображения сравнения подписей")
    
docs_path = 'Docs'
predict_path = 'Predicts/Signs'
employee_signs_path = 'Signatures/' 
predicts_signs_path = 'Predicts/Signs'
#predicts_docs_path = 'Predicts/Docs'

os.system('python model.py')
bot.polling()