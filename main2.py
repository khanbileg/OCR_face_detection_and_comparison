import cv2
import numpy as np
import easyocr
import face_recognition
from PIL import Image
import os
import re

# ------------------ OCR Section ------------------ #
def extract_passport_info(image_path):
    """Extract structured passport information from OCR results using intelligent pattern matching"""
    reader = easyocr.Reader(['en'], gpu=False)
    results = reader.readtext(image_path, detail=1)
    
    passport_info = {
        'passport_number': None,
        'surname': None,
        'given_names': None,
        'nationality': None,
        'birth_date': None,
        'sex': None,
        'issue_date': None,
        'expiry_date': None,
        'authority': None
    }
    
    # Convert results to text for easier searching
    all_text = ' '.join([text for _, text, _ in results])
    
    # Extract passport number (look for patterns like PE0000000)
    passport_pattern = r'[A-Z]{2}\d{7}[A-Z]'
    passport_match = re.search(passport_pattern, all_text)
    if passport_match:
        passport_info['passport_number'] = passport_match.group()
    else:
        # Try alternative pattern for passport numbers
        alt_pattern = r'[A-Z]{2}\d{6,8}[A-Z]'
        alt_match = re.search(alt_pattern, all_text)
        if alt_match:
            passport_info['passport_number'] = alt_match.group()
    
    # Extract dates (look for DD/MM/YYYY or DD MMM YYYY patterns)
    date_patterns = [
        r'\d{2}/\d{2}/\d{4}',  # DD/MM/YYYY
        r'\d{2}\s+[A-Z]{3}\s+\d{4}',  # DD MMM YYYY
    ]
    
    dates_found = []
    for pattern in date_patterns:
        dates = re.findall(pattern, all_text)
        for d in dates:
            if re.match(r'\d{2}\s+[A-Z]{3}\s+\d{4}', d):
                dates_found.append(convert_month_date(d))
            else:
                dates_found.append(d)

    if len(dates_found) >= 2:
        passport_info['birth_date'] = dates_found[0]
        passport_info['issue_date'] = dates_found[1]
        if len(dates_found) >= 3:
            passport_info['expiry_date'] = dates_found[2]
    
    # More intelligent name and information extraction
    potential_names = []
    potential_nationality = None
    potential_sex = None
    potential_authority = []
    name_context = {}  # Store context around names
    
    for i, (bbox, text, conf) in enumerate(results):
        text_upper = text.upper().strip()
        text_clean = text.strip()
        
        # Skip common passport labels and short words
        skip_words = ['PASSPORT', 'TYPE', 'NATIONALITY', 'AUTHORITY', 'EXPIRY', 'ISSUE', 'BIRTH', 'DATE', 'SEX', 
                     'PE', 'MNG', 'PASSPORT NO', 'COUNTRY CODE', 'GIVEN NAMES', 'SURNAME', 'PERSONAL NO']
        
        # Intelligent nationality detection using context clues
        context_words = []
        for j in range(max(0, i-3), min(len(results), i+4)):
            if j != i:
                context_words.append(results[j][1].upper().strip())
        
        # Check if this word appears near "NATIONALITY" or has country-like characteristics
        if (text_clean.isupper() and len(text_clean) >= 3 and 
            not any(char.isdigit() for char in text_clean) and
            (('NATIONALITY' in context_words or 
             any('NATIONALITY' in word for word in context_words)) or
             ('COUNTRY' in context_words or any('COUNTRY' in word for word in context_words)))):
            potential_nationality = text_clean
        
        # Sex detection - look for M/F near "SEX" or "Sex"
        if ('SEX' in context_words or any('SEX' in word for word in context_words)):
            if text_clean.upper() in ['M', 'F', 'MALE', 'FEMALE']:
                potential_sex = text_clean.upper()
        
        # Authority detection - look for words near "AUTHORITY" or "Authority"
        if ('AUTHORITY' in context_words or any('AUTHORITY' in word for word in context_words)):
            if (text_clean.isupper() and len(text_clean) >= 3 and 
                not any(char.isdigit() for char in text_clean)):
                potential_authority.append(text_clean)
        
        # Look for potential names (all caps, 3+ chars, no numbers, not in skip list)
        if (len(text_clean) >= 3 and 
            text_clean.isupper() and 
            not any(char.isdigit() for char in text_clean) and
            text_clean not in skip_words and
            not any(skip in text_upper for skip in skip_words)):
            
            # Store context (nearby words) to help determine if it's surname or given name
            name_context[text_clean] = {
                'context': context_words,
                'confidence': conf,
                'position': i
            }
            potential_names.append(text_clean)
    
    # Additional nationality detection - look for words that appear after "NATIONALITY" label
    for i, (bbox, text, conf) in enumerate(results):
        if 'NATIONALITY' in text.upper():
            # Look for the next few words after "NATIONALITY"
            for j in range(i+1, min(len(results), i+4)):
                next_text = results[j][1].strip()
                if (next_text.isupper() and len(next_text) >= 3 and 
                    not any(char.isdigit() for char in next_text) and
                    next_text not in skip_words):
                    potential_nationality = next_text
                    break
    
    # Also check for standalone country names that appear multiple times
    text_counts = {}
    for _, text, _ in results:
        text_clean = text.strip()
        if (text_clean.isupper() and len(text_clean) >= 3 and 
            not any(char.isdigit() for char in text_clean) and
            text_clean not in skip_words):
            text_counts[text_clean] = text_counts.get(text_clean, 0) + 1
    
    # If a country-like word appears multiple times, it's likely the nationality
    for text, count in text_counts.items():
        if count >= 2 and text not in [passport_info.get('surname'), passport_info.get('given_names')]:
            potential_nationality = text
            break
    
    # Improved sex detection - look for M/F specifically near "SEX" label
    for i, (bbox, text, conf) in enumerate(results):
        text_clean = text.strip()
        if text_clean.upper() in ['M', 'F']:
            # Check if it's near "SEX" or "Sex"
            for j in range(max(0, i-3), min(len(results), i+4)):
                if 'SEX' in results[j][1].upper():
                    potential_sex = text_clean.upper()
                    break
    
    # Additional sex detection - look for sex in context of nearby words
    for i, (bbox, text, conf) in enumerate(results):
        if 'SEX' in text.upper():
            # Look for M/F in the next few words after "SEX"
            for j in range(i+1, min(len(results), i+5)):
                next_text = results[j][1].strip()
                if next_text.upper() in ['M', 'F', 'MALE', 'FEMALE']:
                    potential_sex = next_text.upper()
                    break
                # Also check if M/F is contained within the text
                elif 'M' in next_text.upper() and len(next_text) <= 3:
                    potential_sex = 'M'
                    break
                elif 'F' in next_text.upper() and len(next_text) <= 3:
                    potential_sex = 'F'
                    break
    
    # Filter and assign names intelligently
    if potential_names:
        # Remove duplicates and sort by confidence and length
        unique_names = list(set(potential_names))
        
        # Sort by confidence first, then by length
        unique_names.sort(key=lambda x: (name_context[x]['confidence'], len(x)), reverse=True)
        
        # If we have the same name appearing multiple times, handle it specially
        if len(unique_names) == 1 and potential_names.count(unique_names[0]) >= 2:
            # Same name appears multiple times - likely same first and last name
            passport_info['surname'] = unique_names[0]
            passport_info['given_names'] = unique_names[0]  # Same name for both
            print(f"Note: Same name detected for surname and given names: {unique_names[0]}")
        
        elif len(unique_names) >= 2:
            # Different names - assign by confidence and context
            # Look for context clues to determine which is surname vs given name
            surname_candidates = []
            given_name_candidates = []
            
            for name in unique_names[:2]:  # Take top 2 candidates
                context = name_context[name]['context']
                # Check if name appears near "SURNAME" or "FAMILY NAME" labels
                if any('SURNAME' in word or 'FAMILY' in word for word in context):
                    surname_candidates.append(name)
                # Check if name appears near "GIVEN" or "FIRST" labels
                elif any('GIVEN' in word or 'FIRST' in word for word in context):
                    given_name_candidates.append(name)
                else:
                    # If no clear context, use position and length as fallback
                    if len(name) > 5:  # Longer names are often surnames
                        surname_candidates.append(name)
                    else:
                        given_name_candidates.append(name)
            
            # Assign names based on candidates
            if surname_candidates:
                passport_info['surname'] = surname_candidates[0]
            else:
                passport_info['surname'] = unique_names[0]
                
            if given_name_candidates:
                passport_info['given_names'] = given_name_candidates[0]
            elif len(unique_names) > 1:
                passport_info['given_names'] = unique_names[1]
            
            # Additional logic: if names are similar in length, try to determine by position
            # In passports, surname is often listed first, then given names
            if (passport_info['surname'] and passport_info['given_names'] and
                abs(len(passport_info['surname']) - len(passport_info['given_names'])) <= 2):
                
                # Check positions in the original results
                surname_pos = name_context[passport_info['surname']]['position']
                given_pos = name_context[passport_info['given_names']]['position']
                
                # If surname appears before given name in the text, they might be swapped
                if surname_pos > given_pos:
                    # Swap them - the earlier one is likely the surname
                    temp = passport_info['surname']
                    passport_info['surname'] = passport_info['given_names']
                    passport_info['given_names'] = temp
                    print(f"Note: Swapped names based on position - Surname: {passport_info['surname']}, Given: {passport_info['given_names']}")
            
            # Check if names are very similar (might be same name with slight OCR differences)
            if (passport_info['surname'] and passport_info['given_names'] and
                len(passport_info['surname']) == len(passport_info['given_names']) and 
                sum(1 for a, b in zip(passport_info['surname'], passport_info['given_names']) if a == b) >= len(passport_info['surname']) * 0.8):
                print(f"Note: Similar names detected - possible OCR variation: {passport_info['surname']} vs {passport_info['given_names']}")
        
        elif len(unique_names) == 1:
            passport_info['surname'] = unique_names[0]
    
    # Improved nationality detection - look for common nationality patterns
    nationality_keywords = ['NATIONALITY', 'COUNTRY', 'CITIZENSHIP']
    for i, (bbox, text, conf) in enumerate(results):
        text_clean = text.strip()
        if any(keyword in text_clean.upper() for keyword in nationality_keywords):
            # Look for the next few words after nationality keywords
            for j in range(i+1, min(len(results), i+4)):
                next_text = results[j][1].strip()
                if (next_text.isupper() and len(next_text) >= 3 and 
                    not any(char.isdigit() for char in next_text) and
                    next_text not in skip_words and
                    next_text not in [passport_info.get('surname'), passport_info.get('given_names')]):
                    potential_nationality = next_text
                    break
    
    # Assign nationality - make sure it's not the same as a name
    if potential_nationality and potential_nationality not in [passport_info['surname'], passport_info['given_names']]:
        passport_info['nationality'] = potential_nationality
    
    # If nationality is still None, try to find it in the text
    if passport_info['nationality'] is None:
        # Look for common nationality patterns in the text
        for _, text, _ in results:
            text_clean = text.strip()
            # Common nationality patterns (3-letter country codes or full names)
            if (text_clean.isupper() and len(text_clean) >= 3 and 
                not any(char.isdigit() for char in text_clean) and
                text_clean not in skip_words and
                text_clean not in [passport_info.get('surname'), passport_info.get('given_names')]):
                
                # Check if it looks like a nationality (not a name, not a label)
                if text_clean not in ['PASSPORT', 'TYPE', 'AUTHORITY', 'EXPIRY', 'ISSUE', 'BIRTH', 'DATE', 'SEX']:
                    passport_info['nationality'] = text_clean
                    break
    
    # Assign sex
    if potential_sex:
        passport_info['sex'] = potential_sex
    
    # Assign authority
    if potential_authority:
        passport_info['authority'] = ' '.join(potential_authority)
    
    # Manual extraction of passport number from OCR results
    for _, text, _ in results:
        # Look for passport number patterns in the text
        if re.search(r'[A-Z]{2}\d{6,8}[A-Z]', text):
            # Clean up the passport number
            clean_number = re.sub(r'[^A-Z0-9]', '', text.upper())
            if len(clean_number) >= 8:  # Should be at least 8 characters
                passport_info['passport_number'] = clean_number
    
    return passport_info

def convert_month_date(date_str):
    """Convert month abbreviations to numbers"""
    month_map = {
        'JAN': '01', 'FEB': '02', 'MAR': '03', 'APR': '04', 'MAY': '05', 'JUN': '06',
        'JUL': '07', 'AUG': '08', 'SEP': '09', 'OCT': '10', 'NOV': '11', 'DEC': '12'
    }
    match = re.match(r"(\d{2})\s+([A-Z]{3})\s+(\d{4})", date_str.upper())
    if match:
        day, mon, year = match.groups()
        return f"{day}/{month_map.get(mon, '??')}/{year}"
    return date_str

# ------------------ Face Comparison ------------------ #
def get_face_encoding(image_path):
    image = face_recognition.load_image_file(image_path)
    encodings = face_recognition.face_encodings(image)
    return encodings[0] if encodings else None

def capture_live_face_encoding():
    cap = cv2.VideoCapture(0)
    encoding = None

    if not cap.isOpened():
        print("❌ Could not open camera. Please check if camera is available.")
        return None

    print("📷 Looking for a face. Press 's' to capture when ready, 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to read from camera.")
            continue

        # Flip the frame horizontally for a later selfie-view display
        frame = cv2.flip(frame, 1)
        
        face_locations = face_recognition.face_locations(frame)
        
        # Draw rectangle around detected faces
        for top, right, bottom, left in face_locations:
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, "Face Detected", (left, top - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Add instructions to the frame
        cv2.putText(frame, "Press 's' to capture face", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, "Press 'q' to quit", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("Live Camera - Face Detection", frame)

        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('s'):
            if face_locations:
                try:
                    encoding = face_recognition.face_encodings(frame, known_face_locations=face_locations)[0]
                    print("✅ Face captured successfully.")
                    break
                except Exception as e:
                    print(f"❌ Error capturing face: {e}")
                    print("Please try again.")
            else:
                print("⚠️ No face detected. Please position your face in the camera.")
        elif key == ord('q'):
            print("❌ Cancelled by user.")
            break

    cap.release()
    cv2.destroyAllWindows()
    
    if encoding is None:
        print("❌ Failed to capture face encoding.")
    
    return encoding

def compare_faces(encoding1, encoding2):
    if encoding1 is None or encoding2 is None:
        return None, "Face encoding failed for one or both images."

    try:
        distance = np.linalg.norm(encoding1 - encoding2)
        similarity = max(0, 100 - distance * 100)
        return round(similarity, 2), None
    except Exception as e:
        return None, f"Error comparing faces: {e}"

# ------------------ Main Pipeline ------------------ #
if __name__ == "__main__":
    # Get the current directory and construct the correct path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    PASSPORT_IMG_PATH = os.path.join(current_dir, "images", "22.jpg")
    print("Looking for image at:", PASSPORT_IMG_PATH)

    if not os.path.exists(PASSPORT_IMG_PATH):
        print("❌ Passport image not found.")
        print("Available images in images folder:")
        images_dir = os.path.join(current_dir, "images")
        if os.path.exists(images_dir):
            for file in os.listdir(images_dir):
                if file.endswith(('.jpg', '.jpeg', '.png')):
                    print(f"  - {file}")
        exit()

    print("\n====================")
    print("📄 Extracting Passport Info")
    print("====================")
    passport_info = extract_passport_info(PASSPORT_IMG_PATH)
    
    # Print extracted information
    print("\nExtracted Passport Information:")
    print("="*40)
    for key, value in passport_info.items():
        if value is not None:
            print(f"{key.replace('_', ' ').title()}: {value}")

    print("\n====================")
    print("🧍 Passport Face Encoding")
    print("====================")
    passport_encoding = get_face_encoding(PASSPORT_IMG_PATH)
    
    if passport_encoding is None:
        print("❌ Could not extract face from passport image.")
        print("This might be due to:")
        print("  - No face detected in the passport image")
        print("  - Poor image quality")
        print("  - Face is too small or unclear")
    else:
        print("✅ Passport face encoding extracted successfully.")

    print("\n====================")
    print("📸 Live Camera Face Capture")
    print("====================")
    live_encoding = capture_live_face_encoding()

    if live_encoding is None:
        print("❌ Could not capture live face.")
        print("Face comparison will be skipped.")
    else:
        print("\n====================")
        print("🔍 Face Comparison Result")
        print("====================")
        similarity, error = compare_faces(passport_encoding, live_encoding)

        if error:
            print(f"⚠️ {error}")
        else:
            print(f"✅ Face similarity: {similarity}%")
            
            # Provide interpretation of the similarity score
            if similarity >= 80:
                print("🎉 High similarity - Likely the same person")
            elif similarity >= 60:
                print("✅ Good similarity - Probably the same person")
            elif similarity >= 40:
                print("⚠️ Moderate similarity - Uncertain match")
            else:
                print("❌ Low similarity - Likely different people")
