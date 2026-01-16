import { useState, useEffect } from "react";
import { Card, CardContent } from "./components/ui/Card";
import { Bar, Pie, Line } from "react-chartjs-2";
import Chart from "chart.js/auto";
import { FaUser, FaClock, FaChartLine, FaDoorOpen, FaEye, FaSignInAlt, FaHourglassHalf, FaStar } from "react-icons/fa";

export default function App() {
  const [uploadedVideoUrl, setUploadedVideoUrl] = useState(null);
  const [annotatedVideoUrl, setAnnotatedVideoUrl] = useState(null);
  const [videoFileName, setVideoFileName] = useState(null);
  const [summary, setSummary] = useState(null);
  const [storeLog, setStoreLog] = useState({});
  const [framesData, setFramesData] = useState([]);
  const [loading, setLoading] = useState(false);
  const [selectedPerson, setSelectedPerson] = useState(null);

  const handleUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    setLoading(true);
    setUploadedVideoUrl(URL.createObjectURL(file));
    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch("http://127.0.0.1:8000/api/upload-video/", {
        method: "POST",
        body: formData,
      });
      if (!res.ok) {
        const error = await res.json();
        alert("Erreur backend: " + (error.error || "Unknown"));
        setLoading(false);
        return;
      }
      const data = await res.json();
      setVideoFileName(data.video_file);
    } catch (err) {
      console.error(err);
      alert("Erreur réseau ou serveur");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    let intervalId;
    if (videoFileName) {
      intervalId = setInterval(async () => {
        const res = await fetch(
          `http://127.0.0.1:8000/api/check-status/${videoFileName}/`
        );
        if (!res.ok) {
          console.error("Erreur statut:", await res.json());
          return;
        }
        const data = await res.json();
        if (data.status === "completed") {
          clearInterval(intervalId);
          fetchData();
        }
      }, 5000);
    }
    return () => clearInterval(intervalId);
  }, [videoFileName]);

  const fetchData = async () => {
    try {
      setLoading(true);
      const summaryRes = await fetch(
        `http://127.0.0.1:8000/api/video-summary/${videoFileName}/`
      );
      const storeRes = await fetch(
        `http://127.0.0.1:8000/api/store-log/${videoFileName}/`
      );
      const framesRes = await fetch(
        `http://127.0.0.1:8000/api/video-frames/${videoFileName}/`
      );

      if (!summaryRes.ok || !storeRes.ok || !framesRes.ok) {
        alert("Erreur récupération données depuis MongoDB");
        return;
      }

      const summaryData = await summaryRes.json();
      setSummary(summaryData);
      setAnnotatedVideoUrl(`http://127.0.0.1:8000/api/get-annotated-video/processed_${videoFileName}_h264.mp4`);


      const storeLogData = await storeRes.json();
      setStoreLog(storeLogData.tracks || {});

      const framesJson = await framesRes.json();
      setFramesData(framesJson.frames || []);
    } catch (err) {
      console.error(err);
      alert("Erreur réseau ou serveur");
    } finally {
      setLoading(false);
    }
  };

  const timeSpentData = Object.entries(storeLog).map(([trackId, passages]) => {
    let totalTime = 0;
    passages.forEach((p) => {
      if (p.time_out && p.time_in) totalTime += p.time_out - p.time_in;
    });
    return { id: trackId, totalTime: totalTime.toFixed(2) };
  });

  const entrantsByHour = {};
  Object.values(storeLog).forEach((passages) => {
    passages.forEach((p) => {
      if (p.time_in) {
        const hour = new Date(p.time_in * 1000).getHours();
        entrantsByHour[hour] = (entrantsByHour[hour] || 0) + 1;
      }
    });
  });

  const vitrineStats = { lookAndEnter: 0, lookAndPass: 0 };
  Object.values(storeLog).forEach((passages) => {
    passages.forEach((p) => {
      if (p.looking_vitrine) {
        if (p.entered) vitrineStats.lookAndEnter += 1;
        else vitrineStats.lookAndPass += 1;
      }
    });
  });

  const selectedPersonData = selectedPerson ? storeLog[selectedPerson] : null;

  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        minHeight: "100vh",
        width: "100%",
        backgroundColor: "#f8f8f8ff",
        gap: 24,
        padding: 16,
        fontFamily: "Arial, sans-serif",
      }}
    >
      {loading && (
        <div
          style={{
            position: "fixed",
            top: 0,
            left: 0,
            width: "100%",
            height: "100%",
            backgroundColor: "rgba(255,255,255,0.7)",
            display: "flex",
            justifyContent: "center",
            alignItems: "center",
            fontSize: 24,
            color: "#666",
            zIndex: 1000,
            pointerEvents: "none",
          }}
        >
          Analyse en cours… ⏳
        </div>
      )}

      <h1 style={{ fontSize: 28, fontWeight: "bold", color: "#333" }}>
        🎥 Dashboard Vidéo
      </h1>

      <input
        type="file"
        accept="video/*"
        onChange={handleUpload}
        style={{
          padding: 8,
          borderRadius: 8,
          border: "1px solid #ccc",
          backgroundColor: "#fff",
          cursor: "pointer",
          marginBottom: 16,
        }}
      />
      

      {uploadedVideoUrl && (
        <Card
          style={{
            width: "25vw",
            height: "25vw",
            borderRadius: 12,
            boxShadow: "0 6px 15px rgba(0,0,0,0.1)",
          }}
        >
          <CardContent
            style={{ width: "100%", height: "100%", display: "flex", flexDirection: "column", justifyContent: "center", alignItems: "center" }}
          >
            <h3 style={{ marginBottom: 12 }}>🎬 Vidéo originale</h3>
            <video
              src={uploadedVideoUrl}
              controls
              style={{ width: "90%", height: "90%", objectFit: "cover", borderRadius: 12 }}
            />
          </CardContent>
        </Card>
      )}

      {annotatedVideoUrl && (
        <Card
          style={{
            width: "25vw",
            height: "25vw",
            borderRadius: 12,
            boxShadow: "0 6px 15px rgba(0,0,0,0.1)",
            display: "flex",
            flexDirection: "row",
          }}
        >
          <CardContent style={{ flex: 1, display: "flex", flexDirection: "column", justifyContent: "center", alignItems: "center" }}>
            <h3 style={{ marginBottom: 12 }}>🖌 Vidéo annotée</h3>
            <video
              src={annotatedVideoUrl}
              controls
              style={{ width: "90%", height: "90%", objectFit: "cover", borderRadius: 12 }}
            />
          </CardContent>
          <div style={{ padding: 16, display: "flex", flexDirection: "column", gap: 12 }}>
            <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
              <div style={{ width: 20, height: 20, backgroundColor: "#FF4500", borderRadius: 4 }} /> Inside
            </div>
            <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
              <div style={{ width: 20, height: 20, backgroundColor: "#479747", borderRadius: 4 }} /> Approaching
            </div>
            <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
              <div style={{ width: 20, height: 20, backgroundColor: "#1E90FF", borderRadius: 4 }} /> Looking
            </div>
          </div>
        </Card>
      )}

      {annotatedVideoUrl && summary && (
        <>
          <Card
            style={{
              width: "25vw",
              height: "25vw",
              borderRadius: 12,
              boxShadow: "0 6px 15px rgba(0,0,0,0.1)",
            }}
          >
            <CardContent style={{ width: "100%", height: "100%", padding: 16 }}>
              <h2 style={{ fontSize: 22, fontWeight: 600, marginBottom: 12 }}>📊 Statistiques Globales</h2>
              <p><FaChartLine style={{ marginRight: 8, color: "#5e78ecff" }} />Total passants : {summary.stats.nb_total}</p>
              <p><FaClock style={{ marginRight: 8, color: "#385aeeff" }} />Arrêtés : {summary.stats.nb_arretes}</p>
              <p><FaDoorOpen style={{ marginRight: 8, color: "#115b97ff" }} />Entrés : {summary.stats.nb_entres}</p>
              <p><FaEye style={{ marginRight: 8, color: "#0d34e0ff" }} />Taux d’arrêt : {summary.stats.taux_arret.toFixed(1)}%</p>
              <p><FaSignInAlt style={{ marginRight: 8, color: "#0f206bff" }} />Taux d’entrée : {summary.stats.taux_entree.toFixed(1)}%</p>
              <p><FaHourglassHalf style={{ marginRight: 8, color: "#0e163bff" }} />Temps moyen : {summary.stats.temps_moyen_impression.toFixed(2)} sec</p>
              <p><FaStar style={{ marginRight: 8, color: "#0a102cff" }} />Score attraction : {summary.stats.score_attraction.toFixed(2)}</p>
            </CardContent>
          </Card>

          {timeSpentData.length > 0 && (
            <Card
              style={{
                width: "25vw",
                height: "25vw",
                borderRadius: 12,
                boxShadow: "0 6px 15px rgba(0,0,0,0.1)",
              }}
            >
              <CardContent style={{ width: "100%", height: "100%" }}>
                <h3 style={{ marginBottom: 12 }}>⏱ Temps passé par personne</h3>
                <div style={{ width: "100%", height: "80%" }}>
                  <Bar
                    data={{
                      labels: timeSpentData.map((p) => p.id),
                      datasets: [{ label: "Temps (sec)", data: timeSpentData.map((p) => p.totalTime), backgroundColor: "rgba(46, 78, 223, 0.7)" }],
                    }}
                    options={{
                      responsive: true,
                      maintainAspectRatio: false,
                      plugins: { legend: { display: false } },
                      onClick: (evt, elements) => {
                        if (elements.length > 0) {
                          const index = elements[0].index;
                          const selectedId = timeSpentData[index].id;
                          setSelectedPerson(selectedId);
                        }
                      },
                    }}
                  />
                </div>
              </CardContent>
            </Card>
          )}

          {Object.keys(entrantsByHour).length > 0 && (
            <Card
              style={{
                width: "25vw",
                height: "25vw",
                borderRadius: 12,
                boxShadow: "0 6px 15px rgba(0,0,0,0.1)",
              }}
            >
              <CardContent style={{ width: "100%", height: "100%" }}>
                <h3 style={{ marginBottom: 12 }}>🕒 Entrants par heure</h3>
                <div style={{ width: "100%", height: "80%" }}>
                  <Line
                    data={{
                      labels: Object.keys(entrantsByHour),
                      datasets: [{ label: "Nombre d'entrants", data: Object.values(entrantsByHour), backgroundColor: "rgba(33, 150, 243, 0.2)", borderColor: "#2196f3", fill: true, tension: 0.4 }],
                    }}
                    options={{ responsive: true, maintainAspectRatio: false }}
                  />
                </div>
              </CardContent>
            </Card>
          )}

          <Card
            style={{
              width: "25vw",
              height: "25vw",
              borderRadius: 12,
              boxShadow: "0 6px 15px rgba(0,0,0,0.1)",
            }}
          >
            <CardContent style={{ width: "100%", height: "100%" }}>
              <h3 style={{ marginBottom: 12 }}>🛍 Analyse Vitrine</h3>
              <div style={{ width: "100%", height: "80%" }}>
                <Pie
                  data={{
                    labels: ["Regardent et entrent", "Regardent et passent"],
                    datasets: [{ data: [vitrineStats.lookAndEnter, vitrineStats.lookAndPass], backgroundColor: ["#99c8f1ff", "#272ac0ff"] }],
                  }}
                  options={{ responsive: true, maintainAspectRatio: false }}
                />
              </div>
            </CardContent>
          </Card>

          {selectedPerson && selectedPersonData && (
            <div
              style={{
                position: "fixed",
                top: "10%",
                right: "5%",
                width: "400px",
                maxHeight: "80vh",
                overflowY: "auto",
                backgroundColor: "#fff",
                borderRadius: 12,
                boxShadow: "0 8px 20px rgba(0,0,0,0.25)",
                zIndex: 999,
                padding: 16,
              }}
            >
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12 }}>
                <h3 style={{ margin: 0 }}><FaUser style={{ marginRight: 8 }} /> Infos Personne {selectedPerson}</h3>
                <button onClick={() => setSelectedPerson(null)} style={{ background: "none", border: "none", fontSize: 20, cursor: "pointer", color: "#888" }}>❌</button>
              </div>

              {selectedPersonData.map((p, idx) => (
                <div key={idx} style={{ borderBottom: "1px solid #eee", marginBottom: 8, paddingBottom: 8 }}>
                  <p>Passage #{p.passage_number}</p>
                  <p>Entrée : {p.entered ? "Oui" : "Non"}</p>
                  <p>Sortie : {p.exit ? "Oui" : "Non"}</p>
                  <p>Temps à l’intérieur : {p.duration ? p.duration.toFixed(2) : "-"} sec</p>
                  <p>Regard vitrine : {p.looking_vitrine ? "Oui" : "Non"}</p>
                </div>
              ))}
            </div>
          )}
        </>
      )}
    </div>
  );
}
