import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
from datetime import datetime, timedelta

class InsightsGenerator:
    """Generate business insights and recommendations from sales data"""

    def __init__(self, data_processor):
        self.data_processor = data_processor
        self.insights = {}

    def generate_all_insights(self) -> Dict[str, Any]:
        """Generate comprehensive business insights"""
        insights = {}

        # Basic performance insights
        insights['performance'] = self._generate_performance_insights()

        # Product insights
        insights['products'] = self._generate_product_insights()

        # Brand insights
        insights['brands'] = self._generate_brand_insights()

        # Return insights
        insights['returns'] = self._generate_return_insights()

        # Trend insights
        insights['trends'] = self._generate_trend_insights()

        # Size and customer insights
        insights['customer'] = self._generate_customer_insights()

        # Recommendations
        insights['recommendations'] = self._generate_recommendations()

        self.insights = insights
        return insights

    def _generate_performance_insights(self) -> Dict[str, Any]:
        """Generate performance-based insights"""
        insights = self.data_processor.get_insights()

        performance_insights = {
            'summary': f"From {insights['date_range']['start']} to {insights['date_range']['end']}, your store processed {insights['total_sales']:,} sales transactions worth ₹{insights['total_revenue']:,.0f}, with a {insights['return_rate']:.1f}% return rate.",

            'key_metrics': {
                'total_transactions': insights['total_sales'],
                'total_revenue': insights['total_revenue'],
                'net_revenue': insights['net_revenue'],
                'return_rate': insights['return_rate'],
                'unique_products': insights['unique_products'],
                'unique_brands': insights['unique_brands']
            }
        }

        # Performance classification
        if insights['return_rate'] < 5:
            performance_insights['return_assessment'] = "Excellent - Very low return rate indicates high customer satisfaction"
        elif insights['return_rate'] < 10:
            performance_insights['return_assessment'] = "Good - Return rate is within acceptable range"
        elif insights['return_rate'] < 15:
            performance_insights['return_assessment'] = "Fair - Return rate suggests some quality or sizing issues"
        else:
            performance_insights['return_assessment'] = "Poor - High return rate requires immediate attention"

        return performance_insights

    def _generate_product_insights(self) -> Dict[str, Any]:
        """Generate product-specific insights"""
        insights = self.data_processor.get_insights()

        product_insights = {
            'top_performers': [],
            'underperformers': [],
            'diversity': {},
            'seasonal_patterns': {}
        }

        # Top performing products
        if insights['top_products_qty']:
            top_products = insights['top_products_qty'][:5]
            product_insights['top_performers'] = [
                {
                    'product': f"{p['Brand Code']} - {p['Product Code']}",
                    'quantity': p['Qty'],
                    'impact': 'High performer - Consider increasing stock levels'
                }
                for p in top_products
            ]

        # Product diversity analysis
        if hasattr(self.data_processor, 'sales_df') and self.data_processor.sales_df is not None:
            product_types = self.data_processor.sales_df['Product Code'].value_counts()
            product_insights['diversity'] = {
                'total_product_types': len(product_types),
                'most_common_type': product_types.index[0] if len(product_types) > 0 else 'N/A',
                'diversity_score': min(100, len(product_types) * 10)  # Simple diversity metric
            }

        return product_insights

    def _generate_brand_insights(self) -> Dict[str, Any]:
        """Generate brand-specific insights"""
        insights = self.data_processor.get_insights()

        brand_insights = {
            'top_brands': [],
            'brand_performance': {},
            'brand_recommendations': []
        }

        # Top brands analysis
        if insights['top_brands_qty']:
            top_brands = insights['top_brands_qty'][:5]
            brand_insights['top_brands'] = [
                {
                    'brand': b['Brand Code'],
                    'quantity': b['Qty'],
                    'recommendation': 'Strong performer - Consider expanding product range'
                }
                for b in top_brands
            ]

        # Brand concentration analysis
        if hasattr(self.data_processor, 'sales_df') and self.data_processor.sales_df is not None:
            brand_sales = self.data_processor.sales_df.groupby('Brand Code')['Value'].sum()
            total_sales = brand_sales.sum()
            top_brand_share = (brand_sales.max() / total_sales * 100) if total_sales > 0 else 0

            brand_insights['brand_performance'] = {
                'concentration': 'High' if top_brand_share > 40 else 'Medium' if top_brand_share > 20 else 'Low',
                'top_brand_share': round(top_brand_share, 1),
                'recommendation': 'Consider diversifying if concentration is too high' if top_brand_share > 40 else 'Good brand diversity'
            }

        return brand_insights

    def _generate_return_insights(self) -> Dict[str, Any]:
        """Generate return-specific insights"""
        insights = self.data_processor.get_insights()

        return_insights = {
            'return_analysis': {},
            'return_patterns': {},
            'return_recommendations': []
        }

        if insights['return_analysis']:
            # Analyze return patterns
            returns_df = pd.DataFrame(insights['return_analysis'])
            total_returns = returns_df['Qty'].sum()
            total_return_value = returns_df['Value'].sum()

            return_insights['return_analysis'] = {
                'total_returns': total_returns,
                'total_return_value': total_return_value,
                'avg_return_value': total_return_value / total_returns if total_returns > 0 else 0
            }

            # Identify problematic products
            high_return_products = returns_df[returns_df['Qty'] > returns_df['Qty'].quantile(0.75)]
            return_insights['return_patterns'] = {
                'high_return_products': len(high_return_products),
                'problematic_products': high_return_products[['Brand Code', 'Product Code', 'Qty']].to_dict('records')
            }

            # Generate recommendations
            if insights['return_rate'] > 10:
                return_insights['return_recommendations'].append({
                    'priority': 'High',
                    'issue': 'High return rate detected',
                    'recommendation': 'Review product quality and sizing accuracy. Consider quality checks before shipping.',
                    'action_items': [
                        'Analyze return reasons (if available)',
                        'Check sizing charts accuracy',
                        'Review supplier quality standards',
                        'Consider product photography improvements'
                    ]
                })

        return return_insights

    def _generate_trend_insights(self) -> Dict[str, Any]:
        """Generate trend-based insights"""
        insights = self.data_processor.get_insights()

        trend_insights = {
            'sales_trends': {},
            'seasonal_patterns': {},
            'growth_opportunities': []
        }

        if insights['daily_sales_trend']:
            daily_sales = pd.DataFrame(insights['daily_sales_trend'])

            # Calculate trend metrics
            if len(daily_sales) >= 7:
                recent_sales = daily_sales.tail(7)['Value'].mean()
                overall_avg = daily_sales['Value'].mean()
                trend_direction = 'Improving' if recent_sales > overall_avg * 1.1 else 'Declining' if recent_sales < overall_avg * 0.9 else 'Stable'

                trend_insights['sales_trends'] = {
                    'direction': trend_direction,
                    'recent_avg': round(recent_sales, 0),
                    'overall_avg': round(overall_avg, 0),
                    'change_percent': round(((recent_sales - overall_avg) / overall_avg * 100), 1) if overall_avg > 0 else 0
                }

            # Day of week analysis
            if hasattr(self.data_processor, 'df') and self.data_processor.df is not None:
                day_analysis = self.data_processor.df.groupby('Day_of_Week')['Value'].agg(['sum', 'count', 'mean'])
                best_day = day_analysis.loc[day_analysis['sum'].idxmax()]

                trend_insights['seasonal_patterns'] = {
                    'best_day': best_day.name,
                    'best_day_sales': round(best_day['sum'], 0),
                    'recommendation': f"Focus marketing efforts on {best_day.name} when sales are highest"
                }

        return trend_insights

    def _generate_customer_insights(self) -> Dict[str, Any]:
        """Generate customer behavior insights"""
        insights = self.data_processor.get_insights()

        customer_insights = {
            'size_preferences': {},
            'purchase_patterns': {},
            'customer_recommendations': []
        }

        # Size analysis
        if hasattr(self.data_processor, 'sales_df') and self.data_processor.sales_df is not None:
            size_analysis = self.data_processor.sales_df.groupby('Size').agg({
                'Qty': 'sum',
                'Value': 'sum'
            }).sort_values('Qty', ascending=False)

            top_sizes = size_analysis.head(3)
            customer_insights['size_preferences'] = {
                'most_popular_sizes': top_sizes.index.tolist(),
                'recommendation': 'Stock more of popular sizes and consider size-specific promotions'
            }

        # Purchase frequency analysis
        if hasattr(self.data_processor, 'sales_df') and self.data_processor.sales_df is not None:
            customer_analysis = self.data_processor.sales_df.groupby('Customer Code').agg({
                'Bill No': 'nunique',
                'Value': 'sum',
                'Qty': 'sum'
            })

            repeat_customers = len(customer_analysis[customer_analysis['Bill No'] > 1])

            customer_insights['purchase_patterns'] = {
                'repeat_customers': repeat_customers,
                'repeat_customer_rate': round(repeat_customers / len(customer_analysis) * 100, 1) if len(customer_analysis) > 0 else 0,
                'avg_purchase_value': round(customer_analysis['Value'].mean(), 0) if len(customer_analysis) > 0 else 0
            }

            # Generate recommendations based on customer behavior
            if customer_insights['purchase_patterns']['repeat_customer_rate'] < 20:
                customer_insights['customer_recommendations'].append({
                    'priority': 'Medium',
                    'issue': 'Low repeat customer rate',
                    'recommendation': 'Implement customer retention strategies like loyalty programs or personalized offers',
                    'action_items': [
                        'Create loyalty program',
                        'Send personalized recommendations',
                        'Offer exclusive deals for repeat customers',
                        'Improve post-purchase communication'
                    ]
                })

        return customer_insights

    def _generate_recommendations(self) -> List[Dict[str, Any]]:
        """Generate business recommendations"""
        insights = self.data_processor.get_insights()
        recommendations = []

        # Return rate recommendations
        if insights['return_rate'] > 15:
            recommendations.append({
                'category': 'Returns',
                'priority': 'High',
                'title': 'Reduce Return Rate',
                'description': f'Current return rate of {insights["return_rate"]:.1f}% is above industry average',
                'actions': [
                    'Review product quality standards',
                    'Improve sizing accuracy and charts',
                    'Enhance product photography',
                    'Consider quality inspection before shipping'
                ]
            })

        # Product diversity recommendations
        if insights['unique_products'] < 20:
            recommendations.append({
                'category': 'Product Range',
                'priority': 'Medium',
                'title': 'Expand Product Range',
                'description': f'Only {insights["unique_products"]} unique products - consider diversification',
                'actions': [
                    'Research trending products',
                    'Add complementary product lines',
                    'Consider seasonal items',
                    'Partner with new brands'
                ]
            })

        # Brand concentration recommendations
        if insights['top_brands_qty']:
            top_brand_share = insights['top_brands_qty'][0]['Qty'] / sum(b['Qty'] for b in insights['top_brands_qty']) * 100
            if top_brand_share > 50:
                recommendations.append({
                    'category': 'Brand Strategy',
                    'priority': 'Medium',
                    'title': 'Reduce Brand Concentration',
                    'description': f'Top brand accounts for {top_brand_share:.1f}% of sales - diversify to reduce risk',
                    'actions': [
                        'Explore new brand partnerships',
                        'Promote lesser-known brands',
                        'Balance inventory across brands',
                        'Negotiate better terms with top brand'
                    ]
                })

        # Growth recommendations
        if insights['daily_sales_trend']:
            daily_sales = pd.DataFrame(insights['daily_sales_trend'])
            if len(daily_sales) >= 14:
                recent_trend = daily_sales.tail(7)['Value'].mean()
                previous_trend = daily_sales.iloc[-14:-7]['Value'].mean()
                growth_rate = ((recent_trend - previous_trend) / previous_trend * 100) if previous_trend > 0 else 0

                if growth_rate < -10:
                    recommendations.append({
                        'category': 'Growth',
                        'priority': 'High',
                        'title': 'Address Declining Sales',
                        'description': f'Sales declining by {abs(growth_rate):.1f}% - immediate action required',
                        'actions': [
                            'Analyze market conditions',
                            'Review pricing strategy',
                            'Launch promotional campaigns',
                            'Improve customer engagement'
                        ]
                    })

        # Always include positive reinforcement
        if insights['return_rate'] <= 10 and insights['total_revenue'] > 100000:
            recommendations.append({
                'category': 'General',
                'priority': 'Low',
                'title': 'Maintain Current Strategy',
                'description': 'Your business metrics are healthy - continue current successful practices',
                'actions': [
                    'Monitor key metrics regularly',
                    'Continue successful promotional activities',
                    'Maintain quality standards',
                    'Consider gradual expansion'
                ]
            })

        return recommendations

    def get_formatted_insights(self) -> str:
        """Get formatted insights for display"""
        if not self.insights:
            self.generate_all_insights()

        formatted = []

        # Performance summary
        perf = self.insights['performance']
        formatted.append("📊 **Performance Summary**")
        formatted.append(perf['summary'])
        formatted.append(f"**Return Assessment:** {perf['return_assessment']}\n")

        # Key recommendations
        formatted.append("🎯 **Key Recommendations**")
        for rec in self.insights['recommendations'][:3]:
            formatted.append(f"**{rec['priority']} Priority - {rec['title']}:** {rec['description']}")

        # Top insights
        formatted.append("\n🔍 **Top Insights**")

        # Brand insights
        brand = self.insights['brands']
        if brand['top_brands']:
            formatted.append(f"**Top Brand:** {brand['top_brands'][0]['brand']} with {brand['top_brands'][0]['quantity']:,} units sold")

        # Product insights
        product = self.insights['products']
        if product['top_performers']:
            formatted.append(f"**Best Product:** {product['top_performers'][0]['product']} with {product['top_performers'][0]['quantity']:,} units")

        # Return insights
        returns = self.insights['returns']
        if returns['return_analysis']:
            # Calculate current return rate from the actual data
            total_sales_qty = self.data_processor.sales_df['Qty'].sum() if self.data_processor.sales_df is not None else 0
            total_return_qty = abs(self.data_processor.returns_df['Qty'].sum()) if self.data_processor.returns_df is not None else 0
            current_return_rate = (total_return_qty / total_sales_qty * 100) if total_sales_qty > 0 else 0
            formatted.append(f"**Return Rate:** {current_return_rate:.1f}% - {returns['return_analysis']['total_returns']:,} total returns")

        return "\n".join(formatted)

# Import the data processor for type hints
# from app import DataProcessor  # This will be imported at runtime to avoid circular imports
