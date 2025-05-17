import { HashRouter, Routes, Route } from 'react-router-dom';
import TradePage from './pages/TradePage';
import Dashboard from './pages/Dashboard';

function App() {
  return (
    <HashRouter>
      <Routes>
        <Route path="/" element={<Dashboard />} />
        <Route path="/trade/:symbol" element={<TradePage />} />
      </Routes>
    </HashRouter>
  );
}
