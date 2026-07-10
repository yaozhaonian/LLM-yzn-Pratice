// 组件自动化测试代码，使用 @testing-library/react 测试库，用来验证页面是否正常渲染指定文本。
import { render, screen } from '@testing-library/react';
import App from './App';

test('renders learn react link', () => {
  render(<App />);
  const linkElement = screen.getByText(/learn react/i);
  expect(linkElement).toBeInTheDocument();
});
