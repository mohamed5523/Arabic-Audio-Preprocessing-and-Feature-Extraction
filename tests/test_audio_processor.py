"""
Tests for the Arabic-Audio-Preprocessing-and-Feature-Extraction
"""

import os
import sys
import unittest
import tempfile
import shutil
from unittest.mock import Mock, patch

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import Config
from utils.utils import find_audio_files, validate_audio_file, get_audio_name
from processing.preprocessing import preprocess_audio
from processing.feature_extraction import extract_logmel_features

class TestConfig(unittest.TestCase):
    """Test configuration settings"""
    
    def test_config_values(self):
        """Test that config values are set correctly"""
        self.assertEqual(Config.SAMPLE_RATE, 16000)
        self.assertEqual(Config.VAD_THRESHOLD, 0.99)
        self.assertEqual(Config.EMBEDDING_DIM, 512)
        self.assertEqual(Config.LOGMEL_DIM, 192)
    
    def test_huggingface_token(self):
        """Test Hugging Face token retrieval"""
        token = Config.get_huggingface_token()
        self.assertIsInstance(token, str)
        self.assertTrue(len(token) > 0)

class TestUtils(unittest.TestCase):
    """Test utility functions"""
    
    def setUp(self):
        """Set up test environment"""
        self.test_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up test environment"""
        shutil.rmtree(self.test_dir)
    
    def test_get_audio_name(self):
        """Test audio name extraction"""
        audio_path = "/path/to/audio/test_file.wav"
        name = get_audio_name(audio_path)
        self.assertEqual(name, "test_file")
    
    def test_find_audio_files_empty_folder(self):
        """Test finding audio files in empty folder"""
        files = find_audio_files(self.test_dir)
        self.assertEqual(len(files), 0)
    
    def test_validate_audio_file_nonexistent(self):
        """Test validation of non-existent audio file"""
        is_valid, message = validate_audio_file("nonexistent.wav")
        self.assertFalse(is_valid)
        self.assertIn("not found", message)

class TestFeatureExtraction(unittest.TestCase):
    """Test feature extraction functions"""
    
    def setUp(self):
        """Set up test environment"""
        self.test_dir = tempfile.mkdtemp()
        # Create a dummy audio file for testing
        self.audio_path = os.path.join(self.test_dir, "test.wav")
        
    def tearDown(self):
        """Clean up test environment"""
        shutil.rmtree(self.test_dir)
    
    @unittest.skip("Requires actual audio file")
    def test_extract_logmel_features(self):
        """Test log-mel feature extraction"""
        # This test would require a real audio file
        # In practice, you would create a small test audio file
        features = extract_logmel_features(self.audio_path, "test_audio")
        self.assertIsNotNone(features)
        self.assertIn("logmel_vector", features)

class TestMocking(unittest.TestCase):
    """Test with mocked dependencies"""
    
    @patch('models.models.nemo_asr')
    @patch('models.models.Pipeline')
    @patch('models.models.Model')
    def test_model_manager_init(self, mock_model, mock_pipeline, mock_nemo):
        """Test ModelManager initialization with mocked dependencies"""
        from models.models import ModelManager
        
        # Mock the model loading
        mock_vad_model = Mock()
        mock_nemo.models.EncDecFrameClassificationModel.from_pretrained.return_value = mock_vad_model
        
        mock_diarization = Mock()
        mock_pipeline.from_pretrained.return_value = mock_diarization
        
        mock_embedding = Mock()
        mock_model.from_pretrained.return_value = mock_embedding
        
        # Initialize ModelManager
        manager = ModelManager("fake_token")
        
        # Verify models were loaded
        self.assertIsNotNone(manager.vad_model)
        self.assertIsNotNone(manager.diarization_pipeline)
        self.assertIsNotNone(manager.embedding_model)

class TestIntegration(unittest.TestCase):
    """Integration tests"""
    
    def setUp(self):
        """Set up test environment"""
        self.test_input_dir = tempfile.mkdtemp()
        self.test_output_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up test environment"""
        shutil.rmtree(self.test_input_dir)
        shutil.rmtree(self.test_output_dir)
    
    @unittest.skip("Requires full setup including Milvus")
    def test_audio_processor_init(self):
        """Test AudioProcessor initialization"""
        from core.audio_processor import AudioProcessor
        
        processor = AudioProcessor(
            self.test_input_dir,
            self.test_output_dir,
            "fake_token"
        )
        
        self.assertEqual(processor.input_folder, self.test_input_dir)
        self.assertEqual(processor.output_folder, self.test_output_dir)

if __name__ == '__main__':
    # Create test suite
    test_classes = [
        TestConfig,
        TestUtils,
        TestFeatureExtraction,
        TestMocking,
        TestIntegration
    ]
    
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Exit with appropriate code
    sys.exit(0 if result.wasSuccessful() else 1)