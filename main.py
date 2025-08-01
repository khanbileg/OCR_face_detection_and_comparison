import easyocr
import os
import cv2
import numpy as np
from PIL import Image

# Path to the passport image
IMAGE_PATH = '/Users/macbookpro/Downloads/golomt bank intern/algorithms and codes/cursor version/OCR(Optical Character Recognition)/images/22.jpg'  # Update this path as needed

def preprocess_image(image_path):
    """Preprocess image to improve text detection"""
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not read image: {image_path}")
        return None
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(enhanced, (1, 1), 0)
    
    # Save preprocessed image
    preprocessed_path = image_path.replace('.jpg', '_preprocessed.jpg')
    cv2.imwrite(preprocessed_path, blurred)
    
    return preprocessed_path

def extract_text_easyocr(image_path):
    """Extract text using EasyOCR with preprocessing"""
    print(f"Extracting text from: {image_path}")
    
    # First attempt with original image
    reader = easyocr.Reader(['en'])
    results = reader.readtext(image_path)
    
    return results

def convert_month_date(date_str):
    import re
    month_map = {
        'JAN': '01', 'FEB': '02', 'MAR': '03', 'APR': '04', 'MAY': '05', 'JUN': '06',
        'JUL': '07', 'AUG': '08', 'SEP': '09', 'OCT': '10', 'NOV': '11', 'DEC': '12'
    }
    match = re.match(r"(\d{2})\s+([A-Z]{3})\s+(\d{4})", date_str.upper())
    if match:
        day, mon, year = match.groups()
        return f"{day}/{month_map.get(mon, '??')}/{year}"
    return date_str

def extract_passport_info(results):
    """Extract structured passport information from OCR results using intelligent pattern matching"""
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
    import re
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
        # Look for words that appear near "NATIONALITY" or are country-like patterns
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
        
        # Also check if it's a standalone country name (appears multiple times or in specific context)
        # elif (text_clean.isupper() and len(text_clean) >= 3 and 
        #       not any(char.isdigit() for char in text_clean) and
        #       text_clean not in skip_words and
        #       any('COUNTRY' in word for word in context_words)):
        #     potential_nationality = text_clean
        
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
    
    # Additional sex detection - look for M/F anywhere in the text
    for i, (bbox, text, conf) in enumerate(results):
        text_clean = text.strip()
        if text_clean.upper() in ['M', 'F']:
            # Check if it's near "SEX" or "Sex"
            for j in range(max(0, i-3), min(len(results), i+4)):
                if 'SEX' in results[j][1].upper():
                    potential_sex = text_clean.upper()
                    break
    
    # Also look for sex in context of nearby words
    # for i, (bbox, text, conf) in enumerate(results):
    #     if 'SEX' in text.upper():
    #         # Look for M/F in the next few words after "SEX"
    #         for j in range(i+1, min(len(results), i+5)):
    #             next_text = results[j][1].strip()
    #             if next_text.upper() in ['M', 'F', 'MALE', 'FEMALE']:
    #                 potential_sex = next_text.upper()
    #                 break
    #             # Also check if M/F is contained within the text
    #             elif 'M' in next_text.upper():
    #                 # Extract just the M if it's part of a short text
    #                 potential_sex = 'M'
    #                 break
    #             elif 'F' in next_text.upper():
    #                 # Extract just the F if it's part of a short text
    #                 potential_sex = 'F'
    #                 break
    
    # Broader search for M/F anywhere in the results
    for i, (bbox, text, conf) in enumerate(results):
        text_clean = text.strip()
        if text_clean.upper() in ['M', 'F']:
            # Only assign if it's not part of a country code or other context
            if not any(keyword in ' '.join([results[j][1] for j in range(max(0, i-2), min(len(results), i+3))]).upper() 
                      for keyword in ['COUNTRY', 'CODE', 'MNG', 'USA', 'UK']):
                potential_sex = text_clean.upper()
        elif 'M' in text_clean.upper() and len(text_clean) <= 5:
            # Only assign if it's not part of a country code or other context
            if not any(keyword in ' '.join([results[j][1] for j in range(max(0, i-2), min(len(results), i+3))]).upper() 
                      for keyword in ['COUNTRY', 'CODE', 'MNG', 'USA', 'UK']):
                potential_sex = 'M'
        elif 'F' in text_clean.upper() and len(text_clean) <= 5:
            # Only assign if it's not part of a country code or other context
            if not any(keyword in ' '.join([results[j][1] for j in range(max(0, i-2), min(len(results), i+3))]).upper() 
                      for keyword in ['COUNTRY', 'CODE', 'MNG', 'USA', 'UK']):
                potential_sex = 'F'
    
    # Additional authority detection - look for authority-related phrases
    authority_phrases = []
    for i, (bbox, text, conf) in enumerate(results):
        if any(keyword in text.upper() for keyword in ['GENERAL', 'AUTHORITY', 'STATE', 'REGISTRATION']):
            authority_phrases.append(text.strip())
    
    # Filter and assign names intelligently
    if potential_names:
        # Remove duplicates and sort by length (longer names are more likely to be real names)
        unique_names = list(set(potential_names))
        unique_names.sort(key=len, reverse=True)
        
        # If we have the same name appearing multiple times, handle it specially
        if len(unique_names) == 1 and potential_names.count(unique_names[0]) >= 2:
            # Same name appears multiple times - likely same first and last name
            passport_info['surname'] = unique_names[0]
            passport_info['given_names'] = unique_names[0]  # Same name for both
            print(f"Note: Same name detected for surname and given names: {unique_names[0]}")
        
        elif len(unique_names) >= 2:
            # Different names - assign longest as surname, second longest as given names
            passport_info['surname'] = unique_names[0]
            passport_info['given_names'] = unique_names[1]
            
            # Check if names are very similar (might be same name with slight OCR differences)
            if (len(unique_names[0]) == len(unique_names[1]) and 
                sum(1 for a, b in zip(unique_names[0], unique_names[1]) if a == b) >= len(unique_names[0]) * 0.8):
                print(f"Note: Similar names detected - possible OCR variation: {unique_names[0]} vs {unique_names[1]}")
        
        elif len(unique_names) == 1:
            passport_info['surname'] = unique_names[0]
    
    # Assign nationality - make sure it's not the same as a name
    if potential_nationality and potential_nationality not in [passport_info['surname'], passport_info['given_names']]:
        passport_info['nationality'] = potential_nationality
    
    # Assign sex
    if potential_sex:
        passport_info['sex'] = potential_sex
    
    # Assign authority
    if authority_phrases:
        passport_info['authority'] = ' '.join(authority_phrases)
    
    # Manual extraction of passport number from OCR results
    for _, text, _ in results:
        # Look for passport number patterns in the text
        if re.search(r'[A-Z]{2}\d{6,8}[A-Z]', text):
            # Clean up the passport number
            clean_number = re.sub(r'[^A-Z0-9]', '', text.upper())
            if len(clean_number) >= 8:  # Should be at least 8 characters
                passport_info['passport_number'] = clean_number
    
    return passport_info

if __name__ == "__main__":
    if not os.path.exists(IMAGE_PATH):
        print(f"Image file not found: {IMAGE_PATH}")
    else:
        # Extract all text using EasyOCR
        results = extract_text_easyocr(IMAGE_PATH)
        IMAGE_PATH = preprocess_image(IMAGE_PATH)
        results2 = extract_text_easyocr(IMAGE_PATH)
        
        # Extract structured passport information
        print("\n" + "="*50)
        print("STRUCTURED PASSPORT INFORMATION")
        print("="*50)
        passport_info = extract_passport_info(results)
        passport_info2 = extract_passport_info(results2)
        
        for key, value in passport_info.items():
            for key2, value2 in passport_info2.items():
                if key == key2:
                    if value2 is None:
                        print(f"{key.replace('_', ' ').title()}: {value}")
                    else:
                        print(f"{key.replace('_', ' ').title()}: {value2}")
