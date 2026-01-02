import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import { FileCheck } from 'lucide-react';

const NavItem = ({ to, icon: Icon, label }) => {
  const loc = useLocation();
  const active = loc.pathname === to;
  return (
    <Link
      to={to}
      className={`flex items-center p-3 text-gray-700 hover:bg-gray-100 ${
        active ? 'bg-gray-100 font-semibold' : ''
      }`}
    >
      <Icon className="mr-3" /> {label}
    </Link>
  );
};

const Sidebar = ({ width }) => {
  return (
    <div className="bg-white shadow-lg" style={{ width: `${width}px` }}>
      <div className="p-5">
        <h1 className="text-xl font-bold mb-2">Fin Term Std</h1>
        <p className="text-sm text-gray-600 mb-4">金融专有名词标准化（FAISS 本地索引）</p>
      </div>
      <nav className="mt-2">
        <NavItem to="/stand" icon={FileCheck} label="金融术语标准化" />
      </nav>
      <div className="p-5 text-xs text-gray-500 border-t mt-4">
        后端默认：{process.env.REACT_APP_API_BASE_URL || 'http://127.0.0.1:8009'}
      </div>
    </div>
  );
};

export default Sidebar;
