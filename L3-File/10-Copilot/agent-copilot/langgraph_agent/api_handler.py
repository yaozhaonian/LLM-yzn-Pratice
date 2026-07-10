import json
import requests
from typing import Dict, Any, Optional, List
from utils import logger


class ApiHandler:
    def __init__(self, base_url: str = "http://127.0.0.1:8081", api_key: str = "hihachengfeng"):
        self.base_url = base_url
        self.api_key = api_key
        self.headers = {"X-API_Key": api_key}
    
    def _request(self, method: str, path: str, params: Optional[Dict[str, Any]] = None, 
                 body: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        url = f"{self.base_url}{path}"
        headers = self.headers.copy()
        
        if body:
            headers["Content-Type"] = "application/json"
        
        try:
            logger.info(f"[API调用] {method} {url} params={params} body={body}")
            
            if method.upper() == "GET":
                response = requests.get(url, params=params, headers=headers)
            elif method.upper() == "POST":
                response = requests.post(url, params=params, json=body, headers=headers)
            elif method.upper() == "PUT":
                response = requests.put(url, params=params, json=body, headers=headers)
            elif method.upper() == "DELETE":
                response = requests.delete(url, params=params, json=body, headers=headers)
            else:
                return {"success": False, "error": f"不支持的方法: {method}"}
            
            if response.status_code == 200:
                try:
                    result = response.json()
                    logger.info(f"[API调用成功] {url} -> {json.dumps(result)[:200]}...")
                    return {"success": True, "data": result}
                except json.JSONDecodeError:
                    return {"success": False, "error": "响应不是有效的JSON"}
            else:
                logger.error(f"[API调用失败] {url} -> {response.status_code}")
                return {"success": False, "error": f"HTTP {response.status_code}", "status_code": response.status_code}
        
        except requests.exceptions.RequestException as e:
            logger.error(f"[API调用异常] {url} -> {e}")
            return {"success": False, "error": str(e)}

    def get_product_by_name(self, name: str) -> Dict[str, Any]:
        return self._request("POST", "/products/getProductByName", body={"name": name})
    
    def get_product_by_id(self, product_id: int) -> Dict[str, Any]:
        return self._request("GET", "/products/getProductById", params={"productId": product_id})
    
    def get_batch_products(self, product_ids: List[int]) -> Dict[str, Any]:
        return self._request("POST", "/products/getBatchProductByProductIds", body={"productIds": product_ids})
    
    def add_product(self, name: str, description: str = "", price: float = 0.0, 
                    quantity_in_stock: int = 0) -> Dict[str, Any]:
        return self._request("POST", "/products/addProduct", body={
            "name": name,
            "description": description,
            "price": price,
            "quantityInStock": quantity_in_stock
        })
    
    def update_product_description(self, product_id: int, description: str) -> Dict[str, Any]:
        return self._request("POST", "/products/updateProductDescription", body={
            "productId": product_id,
            "description": description
        })
    
    def get_product_substitutes(self, product_id: int) -> Dict[str, Any]:
        return self._request("GET", "/products/getProductSubstitutes", params={"productId": product_id})
    
    def remove_product_by_name(self, name: str) -> Dict[str, Any]:
        return self._request("DELETE", "/products/removeProductByName", body={"name": name})
    
    def remove_product_by_id(self, product_id: int) -> Dict[str, Any]:
        return self._request("DELETE", "/products/removeProductById", body={"productId": product_id})

    def get_order_by_id(self, order_id: int) -> Dict[str, Any]:
        return self._request("GET", "/orders/getOrderByOrderId", params={"orderId": order_id})
    
    def get_orders_by_product_id(self, product_id: int) -> Dict[str, Any]:
        return self._request("GET", "/orders/getByProductId", params={"productId": product_id})
    
    def get_orders_by_status(self, status: str) -> Dict[str, Any]:
        return self._request("POST", "/orders/getByOrderStatus", body={"status": status})
    
    def get_orders_by_time_range(self, start_date: str, end_date: str) -> Dict[str, Any]:
        return self._request("POST", "/orders/getByTimeRange", body={"startDate": start_date, "endDate": end_date})
    
    def create_order(self, product_id: int, quantity: int, customer_name: str = "匿名用户") -> Dict[str, Any]:
        return self._request("POST", "/orders/createOrder", body={
            "productId": product_id,
            "quantity": quantity,
            "customerName": customer_name
        })
    
    def update_order_status(self, order_id: int, new_status: str) -> Dict[str, Any]:
        return self._request("PUT", "/orders/updateOrderStatus", body={
            "orderId": order_id,
            "status": new_status
        })
    
    def cancel_order(self, order_id: int) -> Dict[str, Any]:
        return self._request("DELETE", "/orders/cancelOrder", body={"orderId": order_id})

    def get_supplier_by_id(self, supplier_id: int) -> Dict[str, Any]:
        return self._request("GET", "/suppliers/getSupplierById", params={"supplierId": supplier_id})
    
    def get_supplier_by_name(self, name: str) -> Dict[str, Any]:
        return self._request("GET", "/suppliers/getSupplierByName", params={"name": name})
    
    def get_supplier_by_status(self, status: str) -> Dict[str, Any]:
        return self._request("GET", "/suppliers/getSupplierByStatus", params={"status": status})
    
    def add_supplier(self, name: str, phone: str, address: str, 
                     delivery_areas: List[str], rating: float = 0.0, 
                     status: str = "InUse") -> Dict[str, Any]:
        return self._request("POST", "/suppliers/addSuppliers", body={
            "name": name,
            "phone": phone,
            "address": address,
            "deliveryAreas": delivery_areas,
            "rating": rating,
            "status": status
        })
    
    def delete_supplier_by_name(self, name: str) -> Dict[str, Any]:
        return self._request("DELETE", "/suppliers/deleteSupplierByName", body={"name": name})
    
    def delete_supplier_by_id(self, supplier_id: int) -> Dict[str, Any]:
        return self._request("DELETE", "/suppliers/deleteSupplierById", body={"id": supplier_id})
    
    def query_suppliers_by_region(self, region: str) -> Dict[str, Any]:
        return self._request("POST", "/suppliers/querySuppliersByDeliveryRegion", body={"deliveryRegion": region})

    def health_check(self) -> Dict[str, Any]:
        return self._request("GET", "/health")
