const express = require('express');
const bodyParser = require('body-parser');
const { MongoClient } = require('mongodb');
const cors = require('cors');

const app = express();
const PORT = 3001;

app.use(cors());
app.use(bodyParser.json());

const uri = "mongodb+srv://Tejas:Tejasiyer%402003@cluster0.fzlpz3a.mongodb.net/";
const client = new MongoClient(uri, { useNewUrlParser: true, useUnifiedTopology: true });

async function connectToDatabase() {
  try {
    await client.connect();
    console.log('Connected to the database');
  } finally {
    // No need to close the connection immediately
  }
}

connectToDatabase();

// Inside your getViolationsData function
const getViolationsData = async (number_plate) => {
  const database = client.db('Cluster0');
  const collection = database.collection('traffic_data1');

  const result = await collection.findOne({ number_plate });

  return result ? {
    _id: result._id,
    number_plate: result.number_plate,
    red_light: result.red_light,
    helmet: result.helmet,
    speeding: result.speeding,
    triple: result.triple,
  } : null;
};


// ... (your existing imports and middleware)

app.post('/', async (req, res) => {
  const { option, value } = req.body;

  if (option === 'vehicle') {
    try {
      const violationsData = await getViolationsData(value);
      res.json({ violations: violationsData });
    } catch (error) {
      console.error(error);
      res.status(500).json({ error: 'Server Error' });
    }
  } else {
    res.status(400).json({ error: 'Invalid option' });
  }
});

// ... (rest of your existing code)


app.listen(3001, () => {
  console.log('Server is running on port 3001');
});
