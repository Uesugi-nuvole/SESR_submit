import re
import hashlib

def calculate_bin_hash(filename):
    try:
        with open(filename, 'r') as f:
            content = f.read()
        
        # Æ¥ÅäÊý×éÄÚÈÝ
        pattern = r'sesr_x4_qat_xl_consci_bin\[\] = \{([^;]+)\};'
        match = re.search(pattern, content, re.DOTALL)
        
        if match:
            hex_string = match.group(1)
            
            # ¡¾ºËÐÄÐÞ¸´¡¿Ê¹ÓÃÕýÔò±í´ïÊ½£¬Ö»ÌáÈ¡ºÏ·¨µÄÊ®Áù½øÖÆ×Ö·û(0-9, a-f, A-F)
            # Õâ»á×Ô¶¯¹ýÂËµô¶ººÅ¡¢»»ÐÐ·û¡¢ÖÆ±í·û¡¢×¢ÊÍµÈËùÓÐ¸ÉÈÅ×Ö·û
            hex_string = ''.join(re.findall(r'[0-9a-fA-F]', hex_string))
            
            # ¼ì²éÌáÈ¡ºóµÄ×Ö·û´®³¤¶ÈÊÇ·ñÎªÅ¼Êý£¨bytes.fromhex ÒªÇó±ØÐëÊÇÅ¼Êý£©
            if len(hex_string) % 2 != 0:
                print("Error: Extracted hex string has an odd length. Please check the file format.")
                return None

            # ½«Ê®Áù½øÖÆ×Ö·û´®×ª»»Îª×Ö½Ú
            bin_data = bytes.fromhex(hex_string)
            
            # ¼ÆËã SHA-256 ¹þÏ£Öµ
            sha256_hash = hashlib.sha256(bin_data).hexdigest()
            
            print(f"Success! SHA-256 hash of 'sesr_x4_qat_xl_consci_bin':")
            print(sha256_hash)
            return sha256_hash
        else:
            print("Error: 'sesr_x4_qat_xl_consci_bin' array not found in the file.")
            return None

    except FileNotFoundError:
        print(f"Error: File '{filename}' not found. Please ensure it exists.")
        return None
    except Exception as e:
        # ¡¾´ÎÒªÐÞ¸´¡¿Ê¹ÓÃÓ¢ÎÄ±¨´í£¬±ÜÃâ·þÎñÆ÷ÖÕ¶Ë²»Ö§³ÖÖÐÎÄµ¼ÖÂµÄ UnicodeEncodeError
        print(f"An unexpected error occurred: {e}")
        return None

if __name__ == "__main__":
    calculate_bin_hash('sesr_x4_qat_xl_consci.mem.h')
