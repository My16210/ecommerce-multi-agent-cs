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
    -- DataCo 的**原值**，9 个英文状态，中英对照见 order_tools.STATUS_GLOSS：
    -- COMPLETE / CLOSED / PROCESSING / PENDING / PENDING_PAYMENT /
    -- ON_HOLD / PAYMENT_REVIEW / SUSPECTED_FRAUD / CANCELED
    -- （这里曾经写的是手写的五个中文状态，换真实数据后那五个没了。注释比代码
    --   更容易烂掉，所以这一列的取值域**只有一个来源**：order_tools.STATUS_GLOSS。）
    status        VARCHAR(16) NOT NULL,
    amount        REAL        NOT NULL,
    region        VARCHAR(32),
    created_at    VARCHAR(32) NOT NULL,
    -- **全表为空**，而且不是"漏灌了"：DataCo 没有支付时间这一列。
    -- 空值的含义因此是"不知道"，不是"未支付"——这两种说法不能互相顶替。
    paid_at       VARCHAR(32)
);

CREATE TABLE IF NOT EXISTS logistics (
    order_id    VARCHAR(32) PRIMARY KEY,
    -- 这两列**全表为空**，是刻意留的：公开数据集里没有承运商和运单号
    -- （快递公司私有数据）。合成一个假值会让下游"不许引用不存在的事实"
    -- 那道检查一起失效，所以宁可空着，回答里"有的才说"。
    carrier     VARCHAR(32),
    tracking_no VARCHAR(64),
    -- **全表为空**：公开数据里没有发货地。列留着是为了 schema 稳定，
    -- 不是因为"以后会填"——真要填得有数据来源才行。
    from_city   VARCHAR(32),
    -- **这一列其实装着目的地**，不是收件城市：派生时填的是 DataCo 的
    -- `Order City`，而官方字段字典对它的定义是 *Destination city*。
    -- 所以列名 `to_city` 是对的，容易看错的是另一头——别把 `from_city`
    -- 当成"发货地有值、只是没填"，它是真的没有。
    to_city     VARCHAR(32),
    -- Delivery Status 原值，实测只有三种：Advance shipping / Late delivery /
    -- Shipping on time。枚举里还有个 "Shipping canceled"，库里一条都没有——
    -- 因为**已取消的订单压根没有物流行**，不是灌数据漏了。
    status      VARCHAR(32),
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
