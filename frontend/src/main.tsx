import ReactDOM from "react-dom/client";
import { lazy, Suspense } from "react";
import PredictionApp from "./PredictionApp";
import "./styles.css";
const LegacyApp = lazy(() => import("./App"));
ReactDOM.createRoot(document.getElementById("root")!).render(
  new URLSearchParams(window.location.search).has("legacy") ? (
    <Suspense fallback={<p>Loading previous workspace…</p>}>
      <LegacyApp />
    </Suspense>
  ) : (
    <PredictionApp />
  ),
);
