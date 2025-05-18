import {
  BrowserRouter as Router,
  Routes,
  Route,
  Navigate,
} from "react-router-dom";
import Dashboard from "./components/Dashboard"; // Your main component
import NotFound from "./components/NotFound"; // Optional: fallback for undefined routes

function App() {
  return (
    <Router basename="/">
      <Routes>
        <Route path="/" element={<Dashboard />} />
        
        {/* Optional: fallback for undefined routes */}
        <Route path="*" element={<Navigate to="/" />} />
        {/* Or use a custom component: <Route path="*" element={<NotFound />} /> */}
      </Routes>
    </Router>
  );
}

export default App;
