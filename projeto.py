# Importa as bibliotecas necessárias
import cv2
import mediapipe as mp
import numpy as np
import random

# Inicializa o módulo de detecção de mãos do MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)  # Apenas uma mão será detectada
mp_draw = mp.solutions.drawing_utils  # Utilitário para desenhar a mão detectada

# Captura de vídeo da webcam
cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX  # Fonte para os textos na tela

# Variáveis do jogo
score = 0
high_score = 0
game_started = False
game_over = False
missed = 0
max_missed = 3

# Carrega as imagens com canal alfa (transparência)
image_paths = {
    "maca": cv2.imread("./assets/maca.png", cv2.IMREAD_UNCHANGED),
    "melancia": cv2.imread("./assets/melancia.png", cv2.IMREAD_UNCHANGED),
    "abacaxi": cv2.imread("./assets/abacaxi.png", cv2.IMREAD_UNCHANGED),
    "bomba": cv2.imread("./assets/bomba.png", cv2.IMREAD_UNCHANGED),
    "coracao": cv2.imread("./assets/coracao.png", cv2.IMREAD_UNCHANGED)
}
fruit_names = ["maca", "melancia", "abacaxi"]

# Função que sobrepõe uma imagem com transparência sobre outra
def overlay_image_alpha(img, img_overlay, pos):
    x, y = pos
    overlay = img_overlay[:, :, :3]
    mask = img_overlay[:, :, 3:] / 255.0  # Canal alfa como máscara

    h, w = overlay.shape[:2]
    if y + h > img.shape[0] or x + w > img.shape[1] or x < 0 or y < 0:
        return img  # Evita desenhar fora da tela

    roi = img[y:y+h, x:x+w]
    img[y:y+h, x:x+w] = (1. - mask) * roi + mask * overlay
    return img

# Classe para representar frutas (ou bombas)
class Fruit:
    def __init__(self, is_bomb=False):
        self.x = random.randint(100, 500)
        self.y = 0
        self.speed = random.randint(5, 9)
        self.is_bomb = is_bomb
        self.name = "bomba" if is_bomb else random.choice(fruit_names)
        self.img = cv2.resize(image_paths[self.name], (70, 70), interpolation=cv2.INTER_AREA)

    def move(self):
        self.y += self.speed  # Faz a fruta cair

    def draw(self, img):
        overlay_image_alpha(img, self.img, (self.x, self.y))  # Desenha a fruta

# Lista de frutas e animações de explosão
fruits = [Fruit()]
explosion_frames = []

# Função para reiniciar o jogo
def reset_game():
    global score, fruits, explosion_frames, missed
    score = 0
    missed = 0
    fruits = [Fruit()]
    explosion_frames = []

# Loop principal do jogo
while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)  # Espelha a imagem da webcam
    h, w, _ = img.shape
    cx, cy = -1, -1  # Coordenadas do dedo indicador
    hand_closed = False

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    # Se uma mão for detectada
    if results.multi_hand_landmarks:
        handLms = results.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)
        lmList = handLms.landmark
        cx, cy = int(lmList[8].x * w), int(lmList[8].y * h)  # Coordenadas do dedo indicador
        cv2.circle(img, (cx, cy), 15, (0, 255, 0), cv2.FILLED)

        # Verifica se a mão está fechada (indicador perto do polegar)
        thumb_tip = lmList[4]
        distance = ((lmList[8].x - thumb_tip.x)**2 + (lmList[8].y - thumb_tip.y)**2)**0.5
        hand_closed = distance < 0.05

    # Tela de início
    if not game_started:
        cv2.putText(img, "FRUIT NINJA COM GESTOS", (30, 60), font, 1.5, (0, 255, 0), 4)
        cv2.putText(img, "FECHE A MAO PARA INICIAR", (30, 100), font, 1, (255, 255, 255), 3)
        if hand_closed:
            game_started = True
            game_over = False
            reset_game()

    # Tela de game over
    elif game_over:
        cv2.putText(img, "FIM DE JOGO", (150, 170), font, 2, (0, 0, 255), 4)
        cv2.putText(img, f"Pontuacao: {score}", (160, 240), font, 1.5, (255, 255, 255), 3)
        cv2.putText(img, f"Recorde: {high_score}", (200, 310), font, 1, (200, 200, 200), 2)
        cv2.putText(img, "FECHE A MAO PARA REINICIAR", (80, 370), font, 1, (255, 255, 255), 2)
        if hand_closed:
            game_started = False  # Volta para a tela de início

    # Jogo em andamento
    else:
        for fruit in fruits[:]:
            fruit.move()
            fruit.draw(img)

            # Verifica colisão entre a mão e a fruta
            if cx != -1:
                d = ((fruit.x - cx)**2 + (fruit.y - cy)**2)**0.5
                if d < 40:
                    if fruit.is_bomb:
                        game_over = True
                        if score > high_score:
                            high_score = score
                    else:
                        score += 1
                        # Animação de explosão
                        for _ in range(10):
                            offset = (random.randint(-30, 30), random.randint(-30, 30))
                            explosion_frames.append(((fruit.x + offset[0], fruit.y + offset[1]), 10))
                    fruits.remove(fruit)

            # Remove frutas que saem da tela
            if fruit.y > h:
                if not fruit.is_bomb:
                    missed += 1
                    if missed >= max_missed:
                        game_over = True
                        if score > high_score:
                            high_score = score
                fruits.remove(fruit)

        # Desenha as explosões
        for i, (pos, size) in enumerate(explosion_frames):
            cv2.circle(img, pos, size, (0, 255, 255), -1)
        explosion_frames = [(p, s - 1) for p, s in explosion_frames if s > 1]

        # Cria novas frutas aleatoriamente
        if random.random() < 0.02:
            fruits.append(Fruit(is_bomb=random.random() < 0.2))

        # Exibe o placar
        cv2.putText(img, f"Score: {score}", (10, 50), font, 1, (255, 255, 255), 3)

        # Exibe os corações representando as vidas restantes
        heart_img = cv2.resize(image_paths["coracao"], (30, 30), interpolation=cv2.INTER_AREA)
        for i in range(max_missed - missed):
            overlay_image_alpha(img, heart_img, (500 + i * 35, 20))

    # Mostra a janela do jogo
    cv2.imshow("Fruit Ninja com Gestos", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera recursos
cap.release()
cv2.destroyAllWindows()
