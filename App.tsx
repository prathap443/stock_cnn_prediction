import { HashRouter, Routes, Route } from 'react-router-dom';

function App() {
  return (
    <HashRouter>
      <Routes>
        <Route path="/trade/:symbol" element={<TradePage />} />
      </Routes>
    </HashRouter>
  );
}
