import React, { useState } from 'react';
import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';
import Sidebar from './components/Sidebar';
import StdPage from './pages/StdPage';

const WelcomePage = () => {
  return (
    <div className="flex flex-col items-center justify-center h-full">
      <h1 className="text-3xl font-bold text-gray-800 mb-4">金融术语标准化</h1>
      <p className="text-xl text-gray-600">请从左侧菜单选择功能</p>
    </div>
  );
};

const App = () => {
  const [sidebarWidth, setSidebarWidth] = useState(260);

  const handleResize = (e) => {
    const next = Math.max(200, Math.min(420, e.clientX));
    setSidebarWidth(next);
  };

  return (
    <Router>
      <div className="flex h-screen bg-gray-100">
        <Sidebar width={sidebarWidth} />
        <div
          className="w-1 cursor-col-resize bg-gray-300 hover:bg-blue-500"
          onMouseDown={() => {
            document.addEventListener('mousemove', handleResize);
            document.addEventListener(
              'mouseup',
              () => {
                document.removeEventListener('mousemove', handleResize);
              },
              { once: true }
            );
          }}
        />
        <main className="flex-1 overflow-y-auto p-5">
          <Routes>
            <Route path="/" element={<WelcomePage />} />
            <Route path="/stand" element={<StdPage />} />
          </Routes>
        </main>
      </div>
    </Router>
  );
};

export default App;
