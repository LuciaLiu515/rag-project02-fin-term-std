import { render, screen } from '@testing-library/react';
import App from './App';

test('renders sidebar label', () => {
  render(<App />);
  expect(screen.getByText(/Fin Term Std/i)).toBeInTheDocument();
});
