import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import ExperimentList from "@/pages/ExperimentList";
import CreateExperiment from "@/pages/CreateExperiment";
import ExperimentDetail from "@/pages/ExperimentDetail";

export default function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<ExperimentList />} />
        <Route path="/create" element={<CreateExperiment />} />
        <Route path="/experiment/:id" element={<ExperimentDetail />} />
      </Routes>
    </Router>
  );
}
