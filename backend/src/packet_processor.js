const axios = require('axios');

const ML_API_URL = process.env.ML_API_URL || 'http://localhost:5000';
const BACKEND_URL = process.env.BACKEND_URL || 'http://localhost:3001';

/**
 * Process packet features through ML API
 * @param {Object} packetFeatures - Extracted packet features
 * @returns {Promise<Object>} Prediction results
 */
async function processPacket(packetFeatures) {
  try {
    const response = await axios.post(`${ML_API_URL}/predict`, {
      features: packetFeatures
    });
    
    return response.data;
  } catch (error) {
    console.error('ML API error:', error.message);
    return { success: false, error: error.message };
  }
}

/**
 * Send alert to backend
 * @param {Object} alert - Alert data
 */
async function sendAlert(alert) {
  try {
    await axios.post(`${BACKEND_URL}/api/alerts`, alert);
  } catch (error) {
    console.error('Backend API error:', error.message);
  }
}

/**
 * Update backend statistics
 * @param {Object} stats - Statistics data
 */
async function updateStatistics(stats) {
  try {
    await axios.post(`${BACKEND_URL}/api/statistics`, stats);
  } catch (error) {
    console.error('Backend API error:', error.message);
  }
}

module.exports = {
  processPacket,
  sendAlert,
  updateStatistics
};


