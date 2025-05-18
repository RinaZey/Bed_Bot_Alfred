import logging
import json
import random
import os
import nltk
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext, CallbackQueryHandler

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

from sentiment import get_sentiment
from wikipedia_utils import get_wiki_articles
from photo_utils import blur_text, add_text, add_image, apply_filter

TOKEN = ''
nltk.download('punkt')

with open('intents_dataset.json', 'r', encoding='utf-8') as f:
    INTENTS = json.load(f)

with open('product_catalog.json', 'r', encoding='utf-8') as f:
    CATALOG = json.load(f)

with open('music_catalog.json', 'r', encoding='utf-8') as f:
    MUSIC = json.load(f)

with open('movies_catalog.json', 'r', encoding='utf-8') as f:
    MOVIES = json.load(f)

with open('dialogues.txt', 'r', encoding='utf-8') as f:
    DIALOGUES = [d.split('\n') for d in f.read().split('\n\n') if len(d.split('\n')) == 2]

user_data_file = 'user_data.json'
if not os.path.exists(user_data_file):
    with open(user_data_file, 'w', encoding='utf-8') as f:
        json.dump({}, f)
with open(user_data_file, 'r', encoding='utf-8') as f:
    USER_DATA = json.load(f)

FOLLOW_UPS_POS = [
    "А что ты думаешь об этом?",
    "Рад, что могу быть полезен! Нужно что-то ещё?",
    "Если интересно — могу рассказать ещё больше!",
    "Пиши в любое время, я тут!"
]
FOLLOW_UPS_NEG = [
    "Я просто пытаюсь помочь... Хочешь — найду для тебя мем?",
    "Ну не злись, я стараюсь. Может, тебе нужен кофе?",
    "Окей, окей. Если надо, пиши!",
    "Ладно, держусь с достоинством."
]
FOLLOW_UPS_NEUT = [
    "А как тебе такой вариант?",
    "Если что — всегда на связи!",
    "Могу предложить ещё пару идей!",
    "Интересно твоё мнение, расскажи!"
]

def save_user_data():
    with open(user_data_file, 'w', encoding='utf-8') as f:
        json.dump(USER_DATA, f)

# ========== ML-модель ==========
X_text = []
y = []
for intent, intent_data in INTENTS.items():
    for example in intent_data["examples"]:
        X_text.append(example)
        y.append(intent)
vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(3, 3))
X = vectorizer.fit_transform(X_text)
clf = LinearSVC()
clf.fit(X, y)

def classify_intent(replica):
    replica_vec = vectorizer.transform([replica])
    intent = clf.predict(replica_vec)[0]
    best_dist = 1.0
    for example in INTENTS[intent]['examples']:
        dist = nltk.edit_distance(replica.lower(), example.lower()) / max(len(replica), 1)
        if dist < best_dist:
            best_dist = dist
    if best_dist < 0.5:
        return intent
    return None

def get_followup(sentiment):
    if sentiment == 'positive':
        return random.choice(FOLLOW_UPS_POS)
    elif sentiment == 'negative':
        return random.choice(FOLLOW_UPS_NEG)
    else:
        return random.choice(FOLLOW_UPS_NEUT)

def get_answer(intent, user_id, sentiment, username=None):
    answers = INTENTS[intent]['responses']
    answer = random.choice(answers)
    if username:
        answer = answer.replace("{имя}", username)
    return f"{answer}\n\n{get_followup(sentiment)}"

def get_dialogue_answer(text, sentiment):
    best_q, best_a, best_score = None, None, 1.0
    for q, a in DIALOGUES:
        score = nltk.edit_distance(text.lower(), q.lower()) / max(len(q), 1)
        if score < best_score:
            best_score = score
            best_q, best_a = q, a
    if best_score < 0.3:
        return f"{best_a}\n\n{get_followup(sentiment)}"
    return None

def start(update: Update, context: CallbackContext):
    name = update.effective_user.first_name or ""
    update.message.reply_text(
        f"Привет, {name}! Я Альфред — твой цифровой собеседник.\n\nЧем могу быть полезен сегодня?",
        reply_markup=main_keyboard()
    )

def main_keyboard():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("Каталог", callback_data="catalog")],
        [InlineKeyboardButton("Википедия", callback_data="wikipedia")],
        [InlineKeyboardButton("Музыка", callback_data="music"),
         InlineKeyboardButton("Фильмы", callback_data="movies")],
        [InlineKeyboardButton("Фото", callback_data="photo")],
        [InlineKeyboardButton("Напоминание", callback_data="reminder")]
    ])

def handle_text(update: Update, context: CallbackContext):
    user_id = str(update.effective_user.id)
    username = update.effective_user.first_name or ""
    text = update.message.text.strip()
    sentiment = get_sentiment(text)
    USER_DATA.setdefault(user_id, {})
    USER_DATA[user_id]['sentiment'] = sentiment
    save_user_data()
    intent = classify_intent(text)
    if intent:
        answer = get_answer(intent, user_id, sentiment, username)
        update.message.reply_text(answer, reply_markup=main_keyboard())
        return
    answer = get_dialogue_answer(text, sentiment)
    if answer:
        update.message.reply_text(answer, reply_markup=main_keyboard())
        return
    update.message.reply_text(
        "Я тебя не совсем понял( Попробуй переформулировать или выбери действие из меню.",
        reply_markup=main_keyboard()
    )

def handle_callback(update: Update, context: CallbackContext):
    query = update.callback_query
    user_id = str(query.from_user.id)
    data = query.data
    query.answer()
    if data == "catalog":
        keyboard = [[InlineKeyboardButton(cat, callback_data=f"cat_{cat}")] for cat in CATALOG]
        query.edit_message_text("Выбери категорию товара:", reply_markup=InlineKeyboardMarkup(keyboard))
    elif data.startswith("cat_"):
        cat = data[4:]
        subs = CATALOG[cat]
        keyboard = [[InlineKeyboardButton(sub, callback_data=f"sub_{cat}_{sub}")] for sub in subs]
        query.edit_message_text(f"Категория: {cat}\nВыбери подкатегорию:", reply_markup=InlineKeyboardMarkup(keyboard))
    elif data.startswith("sub_"):
        _, cat, sub = data.split("_", 2)
        items = CATALOG[cat][sub]
        for item in items:
            btn = InlineKeyboardMarkup([[InlineKeyboardButton("Подробнее", url=item["link"])]])
            img = item["image"]
            try:
                # Локальный файл
                if os.path.exists(img):
                    with open(img, "rb") as photo:
                        query.message.reply_photo(photo, caption=f'{item["name"]}\n{item["description"]}', reply_markup=btn)
                # Прямая ссылка на изображение
                elif img.startswith("http") and (img.endswith(".jpg") or img.endswith(".jpeg") or img.endswith(".png")):
                    query.message.reply_photo(img, caption=f'{item["name"]}\n{item["description"]}', reply_markup=btn)
                else:
                    query.message.reply_text(f'{item["name"]}\n{item["description"]}', reply_markup=btn)
            except Exception as e:
                print(f"Ошибка при отправке фото: {e}")
                query.message.reply_text(f'{item["name"]}\n{item["description"]}\n(фото недоступно)', reply_markup=btn)
        query.message.reply_text("Выбери действие:", reply_markup=main_keyboard())
    elif data == "music":
        keyboard = [[InlineKeyboardButton(genre, callback_data=f"music_{genre}")] for genre in MUSIC]
        query.edit_message_text("Выбери жанр музыки:", reply_markup=InlineKeyboardMarkup(keyboard))
    elif data.startswith("music_"):
        genre = data[6:]
        recs = "\n".join(MUSIC[genre])
        query.edit_message_text(f"Музыка ({genre}):\n{recs}", reply_markup=main_keyboard())
    elif data == "movies":
        keyboard = [[InlineKeyboardButton(genre, callback_data=f"movies_{genre}")] for genre in MOVIES]
        query.edit_message_text("Выбери жанр фильма:", reply_markup=InlineKeyboardMarkup(keyboard))
    elif data.startswith("movies_"):
        genre = data[7:]
        recs = "\n".join(MOVIES[genre])
        query.edit_message_text(f"Фильмы ({genre}):\n{recs}", reply_markup=main_keyboard())
    elif data == "wikipedia":
        context.user_data["waiting_for_wiki"] = True
        query.edit_message_text("Введи тему для поиска статей в Википедии. Я найду что-то интересное!")
    elif data == "reminder":
        context.user_data["waiting_for_reminder"] = True
        query.edit_message_text("Введи напоминание в формате: 20:00 Позвонить маме")
    elif data == "photo":
        context.user_data["waiting_for_photo"] = True
        query.edit_message_text("Пришли фото, чтобы обработать его.")

def handle_message(update: Update, context: CallbackContext):
    user_id = str(update.effective_user.id)
    username = update.effective_user.first_name or ""
    if context.user_data.get("waiting_for_wiki"):
        articles = get_wiki_articles(update.message.text)
        if not articles:
            update.message.reply_text("Статьи не найдены")
        else:
            for art in articles:
                btn = InlineKeyboardMarkup([[InlineKeyboardButton("Открыть", url=art['url'])]])
                update.message.reply_text(f"{art['title']}\n{art['summary']}...", reply_markup=btn)
        context.user_data["waiting_for_wiki"] = False
        return
    if context.user_data.get("waiting_for_reminder"):
        try:
            time_str, text = update.message.text.split(" ", 1)
            USER_DATA.setdefault(user_id, {}).setdefault("reminders", []).append((time_str, text))
            save_user_data()
            update.message.reply_text(f"Напоминание на {time_str} добавлено, {username}! Не забудь 😉")
        except:
            update.message.reply_text("Некорректный формат! Пример: 20:00 Позвонить маме")
        context.user_data["waiting_for_reminder"] = False
        return
    handle_text(update, context)

def handle_photo(update: Update, context: CallbackContext):
    if context.user_data.get("waiting_for_photo"):
        file = update.message.photo[-1].get_file()
        file_path = f"photo_{update.message.from_user.id}.jpg"
        file.download(file_path)
        keyboard = [
            [InlineKeyboardButton("Замазать текст", callback_data=f"photo_blur_{file_path}")],
            [InlineKeyboardButton("Добавить текст", callback_data=f"photo_addtext_{file_path}")],
            [InlineKeyboardButton("Применить фильтр", callback_data=f"photo_filter_{file_path}")]
        ]
        update.message.reply_text("Что сделать с фото?", reply_markup=InlineKeyboardMarkup(keyboard))
        context.user_data["waiting_for_photo"] = False

def handle_photo_callback(update: Update, context: CallbackContext):
    query = update.callback_query
    data = query.data
    query.answer()
    if data.startswith("photo_blur_"):
        fname = data[len("photo_blur_"):]
        outname = f"blur_{fname}"
        blur_text(fname, outname)
        query.message.reply_photo(open(outname, 'rb'), caption="Текст замазан. Надеюсь, результат тебе понравится!", reply_markup=main_keyboard())
    elif data.startswith("photo_addtext_"):
        fname = data[len("photo_addtext_"):]
        outname = f"text_{fname}"
        add_text(fname, "Пример текста", outname)
        query.message.reply_photo(open(outname, 'rb'), caption="Текст добавлен. Вот такой эксперимент", reply_markup=main_keyboard())
    elif data.startswith("photo_filter_"):
        fname = data[len("photo_filter_"):]
        outname = f"filter_{fname}"
        apply_filter(fname, outname, "BLUR")
        query.message.reply_photo(open(outname, 'rb'), caption="Фильтр применён! Если не понравилось — попробуй другой.", reply_markup=main_keyboard())

def main():
    updater = Updater(TOKEN, use_context=True)
    dp = updater.dispatcher

    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(MessageHandler(Filters.text & (~Filters.command), handle_message))
    dp.add_handler(CallbackQueryHandler(handle_callback, pattern="^(?!photo_).+"))
    dp.add_handler(CallbackQueryHandler(handle_photo_callback, pattern="^photo_"))
    dp.add_handler(MessageHandler(Filters.photo, handle_photo))

    updater.start_polling()
    updater.idle()

if __name__ == "__main__":
    main()
