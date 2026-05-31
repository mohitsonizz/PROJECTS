import cv2
import dlib
import numpy as np
import os
import uuid # Unique file naam banane ke liye

# --- SETUP ---
# Folders check karein (Agar nahi hain toh bana lein)
if not os.path.exists('dataset/Open'): os.makedirs('dataset/Open')
if not os.path.exists('dataset/Closed'): os.makedirs('dataset/Closed')

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

cap = cv2.VideoCapture(0)
if not cap.isOpened() or not cap.read()[0]:
    cap = cv2.VideoCapture(1)

open_count = len(os.listdir('dataset/Open'))
closed_count = len(os.listdir('dataset/Closed'))

# Wahi same cropping logic jo main app mein hai
def crop_eye(frame, landmarks, points):
    x_coords = [landmarks.part(i).x for i in points]
    y_coords = [landmarks.part(i).y for i in points]
    x_min, x_max = max(0, min(x_coords) - 5), max(x_coords) + 5
    y_min, y_max = max(0, min(y_coords) - 5), max(y_coords) + 5
    return frame[y_min:y_max, x_min:x_max], (x_min, y_min, x_max, y_max)

print("--- INSTRUCTIONS ---")
print("1. Camera ke saamne baithein aur lighting achi rakhein.")
print("2. Aankhein KHOL kar 'O' dabayein (Open photo save hogi).")
print("3. Aankhein BAND karke 'C' dabayein (Closed photo save hogi).")
print("4. Band karne ke liye 'Q' dabayein.")
print("Target: Kam se kam 50-50 photos dono ki lein.")
print("--------------------")

while True:
    ret, frame = cap.read()
    if not ret: break
    
    # Copy frame for saving clean images
    save_frame = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)
        
        # Sirf dikhane ke liye rectangle banayein
        try:
            _, l_box = crop_eye(frame, landmarks, range(36, 42))
            _, r_box = crop_eye(frame, landmarks, range(42, 48))
            cv2.rectangle(frame, (l_box[0], l_box[1]), (l_box[2], l_box[3]), (0, 255, 0), 1)
            cv2.rectangle(frame, (r_box[0], r_box[1]), (r_box[2], r_box[3]), (0, 255, 0), 1)
        except:
            pass

    # Counts screen par dikhayein
    cv2.putText(frame, f"Saved Open: {open_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"Saved Closed: {closed_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(frame, "Press 'O' for Open, 'C' for Closed, 'Q' to Quit", (10, frame.shape[0]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    cv2.imshow("Data Capture Tool", frame)
    key = cv2.waitKey(1) & 0xFF

    # --- SAVE LOGIC ---
    if key == ord('o') or key == ord('c'):
        if len(faces) > 0:
            # Clean frame se crop karein
            l_eye_img, _ = crop_eye(save_frame, landmarks, range(36, 42))
            r_eye_img, _ = crop_eye(save_frame, landmarks, range(42, 48))
            
            # Unique filename banayein
            img_name = f"{uuid.uuid4()}.jpg"
            
            if key == ord('o'): # Save Open Eyes
                cv2.imwrite(os.path.join('dataset/Open', f"L_{img_name}"), l_eye_img)
                cv2.imwrite(os.path.join('dataset/Open', f"R_{img_name}"), r_eye_img)
                open_count += 2
                print(f"Saved Open Eyes. Total: {open_count}")
                
            elif key == ord('c'): # Save Closed Eyes
                cv2.imwrite(os.path.join('dataset/Closed', f"L_{img_name}"), l_eye_img)
                cv2.imwrite(os.path.join('dataset/Closed', f"R_{img_name}"), r_eye_img)
                closed_count += 2
                print(f"Saved Closed Eyes. Total: {closed_count}")
        else:
            print("Chehra nahi dikh raha! Photo save nahi hui.")

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()