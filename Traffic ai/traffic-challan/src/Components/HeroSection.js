import React, { useState } from 'react';
import '../App.css';
import './HeroSection.css';
import Navbar from './Navbar';

// ... (import statements and other code)

// ... (import statements and other code)

function HeroSection() {
  const [selectedOption, setSelectedOption] = useState('vehicle');
  const [inputValue, setInputValue] = useState('');
  const [details, setDetails] = useState(null);

  const handleOptionChange = (event) => {
    setSelectedOption(event.target.value);
  };

  const handleInputChange = (event) => {
    setInputValue(event.target.value);
  };

  const handleGetDetails = async (event) => {
    event.preventDefault();
    
    try {
      const response = await fetch('http://localhost:3001/', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ option: selectedOption, value: inputValue }),
      });
  
      if (!response.ok) {
        throw new Error('Server Error');
      }
  
      const data = await response.json();
      console.log('Data from server:', data);
  
      // Assuming the response structure is { violations: {...} }
      const { violations } = data;
  
      // Set the details in the state
      setDetails(violations);
    } catch (error) {
      console.error('Error fetching data:', error.message);
    }
  };
  
  
  
  return (
    <>
    <div className='hero-container'>
      <img src="src\Components\back.jpg" alt="" />

      <div className="form-box">
        <div className='input-data'>
        <h1>Challan Details</h1>

        <div className='underline'></div>
        
        <hr className="separator" />

        <form onSubmit={handleGetDetails}>

          <div className='grad'>
            <h2>Vehicle Number</h2>
            <li>License plate is case sensitive</li>
          </div>
          {selectedOption === 'vehicle' && (
            <div>
              <input
                type="text"
                className='helo'
                placeholder='Vehicle number'
                value={inputValue}
                onChange={handleInputChange}
              />
              
            </div>
            
          )}

          <br />
          <input type="submit" value="Get Details" />
        </form>

        {details && (
          <div className='hl'>
            
            <p>Violations:</p>
            <ul>
              <li>Number Plate: {details.number_plate}</li>
              <li>Red Light: {details.red_light*1500}   Times: {details.red_light}</li>
              <li>Helmet: {details.helmet*1000}   Times: {details.helmet}</li>
              <li>Speeding: {details.speeding*500}  Times: {details.speeding}</li>
              <li>Triple Riding: {details.triple*750}       Times: {details.triple}</li>
              <li>Total Fines: {details.triple*750+details.speeding*500+details.helmet*1000+details.red_light*1500}</li>
              <li>The amount above shown is in Rupees </li>
            </ul>
          </div>
        )}
      </div>
    </div>
    </div>
    </>
  );
}

export default HeroSection;