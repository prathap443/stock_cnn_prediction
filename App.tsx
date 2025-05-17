import { BrowserRouter, Routes, Route } from 'react-router-dom';
import TradePage from './pages/TradePage';

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/trade/:symbol" element={<TradePage />} />
        {/* other routes like <Route path="/" element={<Dashboard />} /> */}
      </Routes>
    </BrowserRouter>
  );
}
