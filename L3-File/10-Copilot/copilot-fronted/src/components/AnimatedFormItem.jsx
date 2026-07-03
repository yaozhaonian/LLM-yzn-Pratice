// 动画表单项
/**
 * 封装的 Ant Design 表单校验动画组件
 * 基于 framer-motion 实现：
 * 表单输入框报错提示文字弹出时带有淡入 + 从上滑下的过渡动画，代替原生生硬直接显示的报错文字。
 */
import { Form } from 'antd';
import { motion } from 'framer-motion';

const errorVariants = {
  hidden: { opacity: 0, y: -10 },
  visible: { opacity: 1, y: 0 }
};

export default function AnimatedFormItem({ children, ...props }) {
  return (
    <Form.Item
      {...props}
      validateStatus={props.help ? 'error' : ''}
      help={
        <motion.div
          initial="hidden"
          animate={props.help ? 'visible' : 'hidden'}
          variants={errorVariants}
        >
          {props.help}
        </motion.div>
      }
    >
      {children}
    </Form.Item>
  );
}
/**
 * 优点
 * 高复用：全局表单统一错误动画，不用每个页面重复写动画代码；
 * 无侵入：完全兼容 antd Form.Item 原生属性，替换成本为 0；
 * 动画流畅：依靠 framer-motion 实现顺滑过渡，体验比原生静态提示更好。
 * 隐性小缺陷
 * 只针对 help 驱动错误状态，如果手动设置 validateStatus="error" 但不传 help，动画不会触发；
 * framer-motion 会生成一层额外 div DOM，极少数场景会轻微影响表单布局间距；
 * 没有自定义动画时长、缓动函数的暴露接口，动画效果固定不可外部调整。
 */