"""
Unit tests for ParkiSense AI Database & Authentication Module
"""

import unittest
import uuid
from src.database import (
    init_db,
    register_user,
    authenticate_user,
    hash_password,
)


class TestDatabaseAuth(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        init_db()

    def test_default_demo_accounts(self):
        neurologist = authenticate_user("dr_smith", "admin123")
        self.assertIsNotNone(neurologist)
        self.assertEqual(neurologist["role"], "Neurologist")

        patient = authenticate_user("patient_john", "patient123")
        self.assertIsNotNone(patient)
        self.assertEqual(patient["role"], "Patient")

    def test_invalid_credentials(self):
        self.assertIsNone(authenticate_user("dr_smith", "wrong_password"))
        self.assertIsNone(authenticate_user("non_existent_user", "password"))

    def test_registration_and_login(self):
        uname = f"test_doc_{uuid.uuid4().hex[:8]}"
        ok, msg = register_user(uname, "secure_pass_123", "Neurologist", "Dr. Test Doc")
        self.assertTrue(ok)

        # Login with newly created user
        user = authenticate_user(uname, "secure_pass_123")
        self.assertIsNotNone(user)
        self.assertEqual(user["username"], uname)
        self.assertEqual(user["role"], "Neurologist")
        self.assertEqual(user["full_name"], "Dr. Test Doc")

    def test_duplicate_registration_rejected(self):
        uname = f"test_dup_{uuid.uuid4().hex[:8]}"
        ok1, _ = register_user(uname, "pass1234", "Patient", "Duplicate Test")
        self.assertTrue(ok1)
        ok2, msg = register_user(uname, "pass1234", "Patient", "Duplicate Test")
        self.assertFalse(ok2)
        self.assertIn("already registered", msg)

    def test_password_hashing_uniqueness(self):
        h1, s1 = hash_password("mypassword")
        h2, s2 = hash_password("mypassword")
        # Different salts ensure different hashes
        self.assertNotEqual(s1, s2)
        self.assertNotEqual(h1, h2)


if __name__ == "__main__":
    unittest.main()
