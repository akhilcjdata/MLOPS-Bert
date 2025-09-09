import unittest
from flask_app.simple_app import app

class ChatHistoryTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.client = app.test_client()
        cls.client.testing = True

    def test_home_page_with_history_link(self):
        """Test that home page includes link to chat history"""
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'<title>Sentiment Analysis</title>', response.data)
        self.assertIn(b'View Previous Chats', response.data)
        self.assertIn(b'/history', response.data)

    def test_history_page_empty(self):
        """Test history page when no chats exist"""
        # Clear any existing session
        with self.client.session_transaction() as sess:
            sess.clear()
            
        response = self.client.get('/history')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Chat History', response.data)
        self.assertIn(b'No chat history yet', response.data)

    def test_prediction_and_history(self):
        """Test that predictions are stored in history"""
        # Clear any existing session
        with self.client.session_transaction() as sess:
            sess.clear()
            
        # Make a prediction
        test_text = "This is a wonderful day and I love it!"
        response = self.client.post('/predict', data={'text': test_text})
        self.assertEqual(response.status_code, 200)
        
        # Check that result shows (should be positive because text > 10 chars)
        self.assertIn(b'Positive', response.data)
        
        # Check history page
        response = self.client.get('/history')
        self.assertEqual(response.status_code, 200)
        self.assertIn(test_text.encode(), response.data)
        self.assertIn(b'Positive Sentiment', response.data)

    def test_multiple_predictions_in_history(self):
        """Test multiple predictions stored in correct order"""
        # Clear any existing session
        with self.client.session_transaction() as sess:
            sess.clear()
            
        # Make multiple predictions
        texts = [
            "Short",  # Should be negative (< 10 chars)
            "This is a much longer text that should be positive"  # Should be positive
        ]
        
        for text in texts:
            response = self.client.post('/predict', data={'text': text})
            self.assertEqual(response.status_code, 200)
        
        # Check history page shows both
        response = self.client.get('/history')
        self.assertEqual(response.status_code, 200)
        
        # Both texts should be in history
        for text in texts:
            self.assertIn(text.encode(), response.data)
        
        # Should show both positive and negative
        self.assertIn(b'Positive Sentiment', response.data)
        self.assertIn(b'Negative Sentiment', response.data)

    def test_clear_history(self):
        """Test clearing chat history"""
        # Clear any existing session
        with self.client.session_transaction() as sess:
            sess.clear()
            
        # Add some history
        self.client.post('/predict', data={'text': 'Test message for history'})
        
        # Verify history exists
        response = self.client.get('/history')
        self.assertIn(b'Test message for history', response.data)
        
        # Clear history
        response = self.client.post('/clear_history')
        self.assertEqual(response.status_code, 302)  # Should redirect
        
        # Verify history is cleared
        response = self.client.get('/history')
        self.assertIn(b'No chat history yet', response.data)
        self.assertNotIn(b'Test message for history', response.data)

if __name__ == '__main__':
    unittest.main()