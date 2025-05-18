import {
  BrowserRouter as Router,
  Routes,
  Route
} from "react-router-dom";
import Dashboard from './Dashboard'; // example

function App() {
  return (
    <Router basename="/">
      <Routes>
        <Route path="/" element={<Dashboard />} />
        {/* Add other routes here */}
      </Routes>
    </Router>
  );
}
