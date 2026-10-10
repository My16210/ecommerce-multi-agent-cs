-- 订单库 schema。**一份 SQLite / MySQL 都能跑**——这不是巧合，是刻意的约束：
-- 演示时默认走 SQLite（零依赖），接真库时换 DSN 即可，schema 不该跟着换。
--
-- 为做到这点的三个写法上的注意：
--   1. 主键用 VARCHAR(32) 而不是 TEXT——MySQL 不允许 TEXT 直接做 PRIMARY KEY
--      （要指前缀长度），VARCHAR 两边都认。
--   2. 不写 AUTOINCREMENT / AUTO_INCREMENT——两个方言关键字不同。退货单号在
--      Python 侧生成（uuid 截断），DDL 里就干净了。
--   3. 时间和金额存 VARCHAR / REAL 而不是 DATETIME / DECIMAL——DECIMAL 在
--      SQLite 里会被当亲和性处理、行为不一致。金额在 demo 量级下用 REAL 够，
--      真上线要换成分（整数）存，别用浮点存钱。

CREATE TABLE IF NOT EXISTS orders (
    order_id      VARCHAR(32) PRIMARY KEY,
    customer_id   VARCHAR(32) NOT NULL,
    customer_name VARCHAR(64),
    status        VARCHAR(16) NOT NULL,   -- 已下单 / 已发货 / 运输中 / 已签收 / 已取消
    amount        REAL        NOT NULL,
    region        VARCHAR(32),
    created_at    VARCHAR(32) NOT NULL,
    paid_at       VARCHAR(32)
);

CREATE TABLE IF NOT EXISTS logistics (
    order_id    VARCHAR(32) PRIMARY KEY,
    carrier     VARCHAR(32),
    tracking_no VARCHAR(64),
    from_city   VARCHAR(32),
    to_city     VARCHAR(32),
    status      VARCHAR(32),              -- 已揽收 / 运输中 / 派送中 / 已签收
    updated_at  VARCHAR(32)
);

-- 退货申请。status 只有三种：待审核 / 已通过 / 已拒绝。
-- **这里没有"自动通过"**：退货是花钱且有争议的操作，模型不该有最终决定权。
-- 表本身不阻止重复申请，去重规则在工具层（见 order_tools.create_return），
-- 因为"什么算重复"是会变的业务规则，不该焊进 DDL。
CREATE TABLE IF NOT EXISTS return_requests (
    return_id  VARCHAR(32) PRIMARY KEY,
    order_id   VARCHAR(32) NOT NULL,
    reason     VARCHAR(255) NOT NULL,
    status     VARCHAR(16)  NOT NULL DEFAULT '待审核',
    created_at VARCHAR(32)  NOT NULL
);
